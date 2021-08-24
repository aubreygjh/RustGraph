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
from torch_geometric.nn import global_max_pool 
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import numpy as np
from scipy.sparse.linalg import eigs, eigsh
class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel,self).__init__()
        self.args = args
        #GNN's hyper-parameters
        self.num_layers = args.num_layers
        self.num_nodes = args.num_nodes
        self.in_channels_gnn = args.in_channels_gnn
        self.hidden_channels_gnn = args.hidden_channels_gnn
        self.out_channels_gnn = args.out_channels_gnn
        #RNN's hyper-parameters
        if self.args.add_ev:
            self.in_channels_rnn = args.out_channels_gnn + args.num_nodes
            self.out_channels_rnn = args.out_channels_gnn + args.num_nodes
        else:
            self.in_channels_rnn = args.out_channels_gnn
            self.out_channels_rnn = args.out_channels_gnn
        #Contrastive Learning's hyper-parameters
        self.n_samples = args.n_samples
        self.timestamp = args.timestamp
        #Network component
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(self.in_channels_gnn, self.hidden_channels_gnn))
        self.convs.append(SAGEConv(self.hidden_channels_gnn, self.out_channels_gnn))
        self.rnn = nn.GRU(
                input_size=self.in_channels_rnn,
                hidden_size=self.out_channels_rnn,
                batch_first=True,
                )
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.softmax = nn.Softmax(dim=0)

    def compute_ev(self,data,normalization=None, is_undirected=False):
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

    def forward(self, data_list):
        seq_len = len(data_list)
        last_l_seq=[]
        for t, data in enumerate(data_list):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i != self.num_layers - 1:
                    x = x.relu()
                    x = F.dropout(x, p=0.5, training=self.training)
            x = global_max_pool(x,batch)
            if self.args.add_ev:
                eigen_vector = self.compute_ev(data).to(self.args.device)
                x = torch.cat((x, eigen_vector),dim=1)    
            last_l_seq.append(x)
        z = torch.stack(last_l_seq)
        z = z.transpose(0,1) #(B, L, C) : (1, 600, 512)
        
        nce = 0
        correct = 0
        cnt = 0
        feat_dim = z.shape[-1]
        neg_dist = int(seq_len/6) #n-steps afterward are selected as negative samples
        end = seq_len - self.n_samples - neg_dist - self.timestamp + 2
        start = int(seq_len/8) if int(seq_len/8) < end else 0

        for t_sample in np.arange(start, end): 
            cnt+=1
            encode_samples = torch.empty((self.timestamp, self.n_samples, feat_dim)).float() #e.g. (12,8,512)
            for i in np.arange(1, self.timestamp+1):
                encode_samples[i-1][0] = z[:,t_sample+i,:].view(feat_dim) #first is postive sample ,(1,1,512)
                for j in np.arange(1, self.n_samples):
                    encode_samples[i-1][j] = z[:,t_sample+i+neg_dist+j-1,:].view(feat_dim) #others are negative samples(selected 48 steps away) (1,7,512)
            encode_samples=encode_samples.to(self.args.device)
            #forward_seq = z[:,:t_sample+1,:] #e.g. (1,t_sample,512)
            forward_seq = z #e.g. (1, 600, 512)
            output, _ = self.rnn(forward_seq, None) #output e.g. (1,600,512) hidden e.g. (1,1,512)
            c_t = output[:,t_sample,:].view(feat_dim) #takes the last of output as c_t e.g. (512)   
            for i in np.arange(0, self.timestamp):
                #total = torch.mm(encode_samples[i], torch.transpose(c_t,0,1))    cosine?
                total = F.cosine_similarity(encode_samples[i], c_t,dim=-1) #e.g. (n_samples)
                correct += torch.sum(torch.eq(torch.argmax(self.softmax(total),dim=0),0))
                nce += self.lsoftmax(total)[0]
        nce /= -1.*cnt*self.timestamp
        acc = 1.*correct.item()/(cnt*self.timestamp)
        return nce, acc, output
        


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(in_channels, hidden_channels)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(hidden_channels, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x
        # return x.log_softmax(dim=-1)     
