#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2021/07/19 17:39:20
@Author  :   Guo Jianhao 
'''
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
from sklearn.metrics import roc_auc_score

from utils import *
from model import MyModel, MLP
from dataset import Elliptic, Digg, UCI


import networkx as nx
import matplotlib.pyplot as plt   
def draw(edge_index, y, name=None):
    G = nx.MultiGraph(node_size=15, font_size=8)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    edgelist = zip(src, dst)
    for i, j in edgelist:
        G.add_edge(i, j)
    plt.figure(figsize=(20, 14)) # 设置画布的大小
    if y == 1:
        nx.draw_networkx(G,node_color="red")
    else:
        nx.draw_networkx(G,node_color="blue")
    plt.savefig('figs/{}.png'.format(name if name else 'path'))
    print(f'Saved fig-{name}.')

if __name__ == '__main__':
    logger = SummaryWriter()
    args = args_parser()
    exp_details(args)

    #Init dataloader
    #data = Elliptic(root='dataset/elliptic_bitcoin_dataset')
    #data = TUDataset(root='dataset/ENZYMES', name='ENZYMES')
    data = []
    if args.dataset == 'uci':
        data = UCI(root='dataset/UCI')
        loader = DataLoader(data, batch_size=1, shuffle=False)
        data =  [data_item.to(args.device) for data_item in loader]
        labels = torch.tensor([data_item['y'].item() for data_item in loader]).to(args.device)
    else:
        exit('Error: Unspecified Dataset')

    # #visualize graphs
    # for i,graph in enumerate(data):
    #     draw(graph.edge_index, graph.y, i)

    start_time = time.time()
    #Init model
    encoder = MyModel(args)
    encoder = encoder.to(args.device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    #Start pre-training
    encoder.train()    
    for epoch in tqdm(range(args.epochs)):
        optimizer.zero_grad()
        #unsupervised
        loss, acc, hidden = encoder(data)
        loss.backward()
        optimizer.step()
        logger.add_scalar('pre-loss', loss.item(), epoch)
        print(f'train_loss: {loss.item():.4f}')
        print(f'train_acc: {acc:.4f}')
    

    # #Start fine-tuning
    # classifier = MLP(256, 2)
    # classifier = classifier.to(args.device)
    # optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=args.weight_decay)
    # # criterion = torch.nn.NLLLoss().to(args.device)
    # criterion = torch.nn.CrossEntropyLoss().to(args.device)
    # classifier.train()
    
    # for epoch in tqdm(range(500)):
    #     optimizer.zero_grad()
    #     with torch.no_grad():
    #         _, _, hidden=encoder(data)
        
    #     hidden = hidden.view(-1, 256)
    #     out, prob = classifier(hidden)   
    #     loss = criterion(out, labels)
    #     loss.backward()
    #     optimizer.step()


    #     y_true = copy.deepcopy(labels).cpu().numpy()
    #     y_score = prob[:,1].cpu().detach().numpy()
    #     auc = roc_auc_score(y_true, y_score)
    #     correct = torch.sum(torch.eq(torch.argmax(prob, dim=1),labels))
    #     acc = 1.*correct/len(y_true)

    #     logger.add_scalar('loss', loss.item(), epoch)
    #     logger.add_scalar('auc', auc, epoch)
    #     logger.add_scalar('acc', acc, epoch)
    #     print(f'Loss: {loss.item():.4f}')
    #     print(f'Auc: {auc:.4f}')
    #     print(f'Acc: {acc:.4f}')
        


    logger.close()
    print('\n Total Training Rime:{0:0.4f}'.format(time.time()-start_time))