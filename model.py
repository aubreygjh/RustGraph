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

class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel,self).__init__()
        self.args = args
        self.num_layers = self.args.num_layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(self.args.in_channels,self.args. hidden_channels))
        self.convs.append(SAGEConv(self.args.hidden_channels, self.args.out_channels))
        self.rnn = nn.GRU(
                input_size=self.args.out_channels,
                hidden_size=self.args.num_classes,
                )
    def forward(self, data_list):
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
        last_l_seq = torch.stack(last_l_seq)


        #input:tensor of  shape (seq_len, batch, feature_dim)
        #output:tensor of shape (seq_len, batch, out_dim)
        out, _ = self.rnn(last_l_seq, None)
        out = F.log_softmax(out.squeeze(1), dim=-1)
        return out
    