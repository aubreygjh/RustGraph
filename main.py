#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2021/07/19 17:39:20
@Author  :   Guo Jianhao 
'''
import os
import time
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score

from utils import *
from model import Model, MLP
from dataset import Elliptic, Digg, UCI


if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    logger = SummaryWriter(logdir='runs/{}_lr{}_epoch{} '.format(time.strftime('%m-%d %H:%M:%S')
                                                        , args.lr
                                                        , args.epochs))
    # exp_details(args)

    #Init dataloader
    dataset = []
    if args.dataset == 'uci':
        dataset = UCI(root='dataset/UCI')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        # labels = torch.tensor([data_item['y'].item() for data_item in loader]).to(args.device)
    elif args.dataset == "elliptic":
        data = Elliptic(root='dataset/elliptic_bitcoin_dataset')
    else:
        exit('Error: Unspecified Dataset')
    nodes = np.load(os.path.join('dataset/UCI/raw', "nodes.npz"), allow_pickle=True)['nodes']
    tensor_nodes = [torch.LongTensor(np.array(node)).to(args.device) for node in nodes]
    adjs = np.load(os.path.join('dataset/UCI/raw', "adjmatrix.npz"), allow_pickle=True)['adj']
    tensor_adjs = [torch.FloatTensor(np.array(adj)).to(args.device) for adj in adjs]
    
    #Init model
    model = Model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    load_time = time.time()
    print('\n Total Loading Rime:{0:0.4f}'.format(load_time-start_time))

    #Start pre-training
    model.train()    
    for epoch in tqdm(range(args.epochs)):
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            nll_loss, kld_loss, jsd_loss, regularizer = model(dataloader, tensor_nodes, tensor_adjs)
            loss = nll_loss + kld_loss + args.lamda * (jsd_loss + args.eps * regularizer)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        logger.add_scalar('nll_loss', nll_loss.item(), epoch)
        logger.add_scalar('kld_loss', kld_loss.item(), epoch)
        logger.add_scalar('jsd_loss', jsd_loss.item(), epoch)
        logger.add_scalar('regularizer', regularizer.item(), epoch)
        logger.add_scalar('loss', loss.item(), epoch)
        print(f'train_loss: {loss.item():.4f}')
    

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
    print('\n Total Training Rime:{0:0.4f}'.format(time.time()-load_time))