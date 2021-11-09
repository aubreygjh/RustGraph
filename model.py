#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/07/19 17:36:57
@Author  :   Guo Jianhao 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GAE, global_max_pool
from torch_geometric.nn.glob.glob import global_mean_pool
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import numpy as np
from scipy.sparse.linalg import eigs, eigsh

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(SAGEConv(input_dim, hidden_dim))
            else:
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))

    def forward(self, data): 
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = self.activation(x)
                x = F.dropout(x, p=0.5, training=self.training)
        g = global_max_pool(x, batch)
        
        return x, g

class GAE(nn.Module):
    def __init__(self, args):
        super().__init__()

class CPC(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, sample_num, timespan):
        super(CPC, self).__init__()
        self.device = device
        self.sample_num = sample_num
        self.timespan = timespan
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.softmax = nn.Softmax(dim=0)
        

    def forward(self, data): #shape: (B, L, C)
        #parameters
        seq_len = data.shape[1]
        feat_dim = data.shape[2]
        neg_dist = int(seq_len/6)
        end = seq_len - self.sample_num - neg_dist - self.timespan + 2
        start = int(seq_len/8) if int(seq_len/8) < end else 0
        cnt = 0
        nce = 0
        correct = 0

        z,_ = self.rnn(data, None) #shape: (B, L, C)
        global_mean = np.squeeze(z.mean(1))
        for t_sample in np.arange(start, end):   #better be [0: len(data))
            cnt+=1
            encode_samples = torch.empty((self.timespan, self.sample_num, feat_dim)).float().to(self.device)
            for i in np.arange(1, self.timespan+1):
                encode_samples[i-1][0] = data[:,t_sample+i,:].view(feat_dim)
                for n_sample in np.arange(1, self.sample_num):
                    encode_samples[i-1][n_sample] = data[:,t_sample+i+neg_dist+n_sample-1,:].view(feat_dim)
            # encode_samples=encode_samples.to(self.args.device)
            c_t = z[:,t_sample,:].view(feat_dim)
            for i in np.arange(0, self.timespan):
                total = F.cosine_similarity(encode_samples[i], c_t, dim=-1)
                nce += self.lsoftmax(total)[0]
                correct += torch.sum(torch.eq(torch.argmax(self.softmax(total),dim=0),0))
        nce /= -1.*cnt*self.timespan
        acc = 1.*correct.item()/(cnt*self.timespan)

        return z, nce, acc

class ENC(nn.Module):
    def __init__(self, args):
        super(ENC, self).__init__()
        self.args = args
        self.gnn_enc = GConv(args.input_dim_gconv, args.hidden_dim_gconv, args.num_layers_gconv).to(args.device)
        self.cpc_enc = CPC(args.device, args.input_dim_rnn, args.hidden_dim_rnn, args.sample_num, args.timespan).to(args.device)
        self.gae_enc = GAE(args).to(args.device)

    def forward(self, dataloader): 
        graph_embed_list = []
        for t, data in enumerate(dataloader):
            data = data.to(self.args.device)
            x, g = self.gnn_enc(data)
            #Use eigenvector as feature, may be deprecated later
            # if self.args.add_ev:    
            #     eigen_vector = self.compute_ev(data).to(self.args.device)
            #     x = torch.cat((x, eigen_vector), dim=1) 
            # g = self.projection(g)
            graph_embed_list.append(g)
        graph_embed_list = torch.stack(graph_embed_list).transpose(0, 1)
        global_mean_pool = torch.squeeze(graph_embed_list.mean(1))
        
        z, nce, acc = self.cpc_enc(graph_embed_list)
        return nce, acc

    def compute_ev(self, data, normalization=None, is_undirected=False):
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'
        edge_weight = data.edge_attr
        if edge_weight is not None and edge_weight.numel() != data.num_edges:
            edge_weight = None

        edge_index, edge_weight = get_laplacian(data.edge_index, edge_weight,
                                                normalization,
                                                num_nodes=data.num_nodes)

        L = to_scipy_sparse_matrix(edge_index, edge_weight, data.num_nodes)

        eig_fn = eigs
        if is_undirected and normalization != 'rw':
            eig_fn = eigsh

        lambda_max,ev = eig_fn(L, k=1, which='LM', return_eigenvectors=True)
        ev = np.squeeze(ev.real)
        ev = np.pad(ev,(0,self.num_nodes-ev.shape[-1]), 'constant', constant_values=(0,0))
        ev = np.reshape(ev,(1,self.num_nodes))
        ev = torch.from_numpy(ev)
        return ev #eigen_vectors of shape(dim, k); dim=length of matrix, k=top-k vectors


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels // 16),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(in_channels // 16, out_channels)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        return x, self.softmax(x)
     
