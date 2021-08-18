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
import numpy as np
class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel,self).__init__()
        self.args = args
        self.num_layers = self.args.num_layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(self.args.in_channels,self.args.hidden_channels))
        self.convs.append(SAGEConv(self.args.hidden_channels, self.args.out_channels))
        self.rnn = nn.GRU(
                input_size=self.args.out_channels,
                hidden_size=self.args.num_classes,
                )
        self.timestamp = 12
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.softmax = nn.Softmax(dim=0)
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
            last_l_seq.append(x)
        z = torch.stack(last_l_seq)
        z = z.transpose(0,1) #(B, L, C) : (1, 600, 3)
        n_samples = 8
        feat_dim = 6
        nce = 0
        correct = 0
        cnt = 0
        neg_dist = int(seq_len/6) #negative sample steps
        for t_sample in np.arange(int(seq_len/6), int(seq_len*5/6)-20): #48 is the distance from c_t to negative samples

            cnt+=1
            encode_samples = torch.empty((self.timestamp, n_samples, feat_dim)).float() #e.g. (12,8,512)
            for i in np.arange(1, self.timestamp+1):
                encode_samples[i-1][0] = z[:,t_sample+i,:].view(feat_dim) #first is postive sample
                for j in np.arange(1, n_samples):
                    encode_samples[i-1][j] = z[:,t_sample+i+neg_dist+j,:].view(feat_dim) #others are negative samples(selected 48 steps away)
            encode_samples=encode_samples.to(self.args.device)
            forward_seq = z[:,:t_sample+1,:] #e.g. (1,100,512)
            output, hidden = self.rnn(forward_seq, None) #output e.g. (1,100,512)
            c_t = output[:,t_sample,:].view(1, feat_dim) #takes the last of output as c_t e.g. (1, 512)
            
            for i in np.arange(0, self.timestamp):
                total = torch.mm(encode_samples[i], torch.transpose(c_t,0,1)) #e.g. (n_samples, 1)   cosine???????
                correct += torch.sum(torch.eq(torch.argmax(self.softmax(total),dim=0),0))
                nce += self.lsoftmax(total)[0]
        nce /= -1.*cnt*self.timestamp
        acc = 1.*correct.item()/(cnt*self.timestamp)
        return nce, acc, hidden
        # t_samples = torch.randint(z.shape()[1]-self.timestamp, size=(1,)).long()
        



    