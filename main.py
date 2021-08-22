#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2021/07/19 17:39:20
@Author  :   Guo Jianhao 
'''

#comment these
from logging import raiseExceptions
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import TUDataset
#comment these

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from tensorboardX import SummaryWriter

from utils import *
from model import MyModel, MLP
from dataset import Elliptic, Digg, UCI

# def process(raw_dir):
#     file_features = os.path.join(raw_dir,'elliptic_txs_features.csv')
#     df = pd.read_csv(file_features, index_col = 0, header=None)
#     xx_time = df.iloc[:,0]-1
#     x_time = torch.from_numpy(xx_time.to_numpy()).long()
#     x = torch.from_numpy(df.to_numpy()).float() 
#     num_nodes = len(df.index.to_numpy())
#     # index_conversion = pd.Series([i for i in range(num_nodes)], index=df.index.to_numpy())
    
   
#     file_classes = os.path.join(raw_dir, 'elliptic_txs_classes.csv')  
#     df = pd.read_csv(file_classes, index_col='txId', na_values='unknown').fillna(0) - 1
#     # print(df.to_numpy())
#     y = torch.from_numpy(df.to_numpy()).view(-1).long()
#     index_conversion = df.reset_index().reset_index().drop(columns='class').set_index('txId')

 
#     file_edges = os.path.join(raw_dir, 'elliptic_txs_edgelist.csv')
#     edge_df = pd.read_csv(file_edges).join(index_conversion, on='txId1', how='inner')
#     edge_df = edge_df.join(index_conversion, on='txId2', how='inner', rsuffix='2').drop(columns=['txId1', 'txId2'])
#     xx_time = xx_time.reset_index().drop(columns=0).reset_index()
#     edge_df = edge_df.join(xx_time, on='index', how='inner', rsuffix='x').drop(columns='indexx')
    

#     data_list = []
#     for i in range(x_time.max().item()+1):
#         node_mask = x_time == i
#         data = Data()
#         data.x = x[node_mask, :]
#         data.y = y[node_mask]

#         edge_index =  torch.from_numpy(edge_df.to_numpy()).t().contiguous()
#         edge_mask = edge_index[2,:] == i
#         edge_index = edge_index[:, edge_mask][:2]
#         data.edge_index = to_undirected(edge_index, num_nodes)

#         # print(data.x)
#         print(data.edge_index)
#         # print(data.y.count_nonzero())
#         # print(data.x.shape)
#         # print(data.y.shape)
#         data_list.append(data)

#     # if self.pre_filter is not None:
#     #     data_list = [d for d in data_list if self.pre_filter(d)]

#     # if self.pre_transform is not None:
#     #     data_list = [self.pre_transform(d) for d in data_list]

#     # data, slices = self.collate(data_list)
#     # torch.save((data, slices), self.processed_paths[0])

# process(raw_dir='dataset/elliptic_bitcoin_dataset/raw')   


if __name__ == '__main__':
    logger = SummaryWriter('logs')
    args = args_parser()
    exp_details(args)

    #Init dataloader
    # data = Elliptic(root='dataset/elliptic_bitcoin_dataset')
    
    #data = TUDataset(root='ENZYMES', name='ENZYMES')
    data = []
    if args.dataset == 'uci':
        data = UCI(root='dataset/UCI')
        loader = DataLoader(data, batch_size=1, shuffle=False)
        data =  [data_item.to(args.device) for data_item in loader]
    else:
        exit('Error: Unspecified Dataset')
    


    #Init model
    encoder = MyModel(args)
    encoder = encoder.to(args.device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #Start pre-training
    start_time = time.time()
    encoder.train()    
    out_log = []
    for epoch in tqdm(range(args.epochs)):
        train_loss = 0.0
        optimizer.zero_grad()
       
        #supervised
        # out = encoder(data)
        # y = torch.empty(0).to(args.device)
        # for data_item in data:
        #     y = torch.cat((y,data_item.y), 0).long()
        # loss = F.nll_loss(out, y)

        #unsupervised
        loss, acc, hidden = encoder(data)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        # out_log.append([F.softmax(out, dim=1), y])
        #evaluate on validation set after every x epoch
        print(f'train_loss: {train_loss:.4f}')
        print(f'train_acc: {acc:.4f}')
    

    #evaluate on test set after completing training

    logger.close()
    print('\n Total Training Rime:{0:0.4f}'.format(time.time()-start_time))