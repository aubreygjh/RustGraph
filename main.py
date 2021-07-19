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
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from tensorboardX import SummaryWriter

from options import *
from utils import *
from model import MyModel

if __name__ == '__main__':
    logger = SummaryWriter('/logs')
    args = args_parser()
    exp_details(args)

    #Init dataloader
    train_data, validation_data, test_data = dataloader(args)
    if args.multi_gpu:
        loader = DataListLoader
    else:
        loader = DataLoader
    train_loader = loader()
    validation_loader = loader()
    test_loader = loader()

    #Init model
    model = MyModel(args)
    if args.multi_gpu:
        model = DataParallel(model)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #Some real-time indices


    #Start training
    start_time = time.time()
    model.train()
    for epoch in tqdm(range(args.epochs)):
        train_loss = 0.0
        out_log = []
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if not args.multi_gpu:
                data = data.to(args.device)
            out = model(data)
            #supervised
            if args.multi_gpu:
                y = torch.cat([d.y.unsqueeze(0) for d in data].squeeze().to(out.device))
            else:
                y = data.y
            loss = F.nll_loss(out, y)
            loss.backward()
            optimize.step()
            train_loss += loss.item()
            out_log.append([F.softmax(out, dim=1), y])
        #evaluate on validation set after every x epoch

        print()
    
    #evaluate on test set after completing training
    print()