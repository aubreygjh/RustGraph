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
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score
from utils import *
from model import  Model
from dataset import  DynamicGraphAnomaly


def initialize():
    #torch.manual_seed(0)
    model = Model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer

if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    start_time = time.time()
    args = args_parser()
    #tensorboard
    tb_path = f'runs/{args.dataset}/{args.anomaly_ratio}/'
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    tb = SummaryWriter(logdir=tb_path+f"{time.strftime('%m-%d,%H:%M:%S')}_snaps{args.snap_size}_train{args.train_ratio}_anomaly{args.anomaly_ratio}_epochs{args.epochs}_lr{args.lr}_xdim{args.x_dim}_hzdim{args.z_dim}_eps{args.eps}_lambda{args.lamda}")
    #log
    log_path = f'logs/{args.dataset}/{args.anomaly_ratio}/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, f'snaps{args.snap_size}_train{args.train_ratio}_anomaly{args.anomaly_ratio}_epochs{args.epochs}_lr{args.lr}_xdim{args.x_dim}_hzdim{args.z_dim}_eps{args.eps}_lambda{args.lamda}')
    log = open(log_file, "a")
    log.writelines(time.strftime('%m-%d %H:%M:%S') + "\n")
    log.writelines(str(args) + "\n")
    #model
    model_path = f'./models/{args.dataset}/{args.anomaly_ratio}/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_file = os.path.join(model_path, f'snaps{args.snap_size}_train{args.train_ratio}_anomaly{args.anomaly_ratio}_epochs{args.epochs}_lr{args.lr}_xdim{args.x_dim}_hzdim{args.z_dim}_eps{args.eps}_lambda{args.lamda}.pkl')

    #Init dataloader
    dataset = []
    if args.dataset not in ['uci', 'digg', 'btc_alpha', 'btc_otc', 'email', 'as_topology', 'hepth']:
        raise NotImplementedError
    dataset = DynamicGraphAnomaly(root='dataset', name=args.dataset, args=args)
    train_size = dataset.train_size
    data_train = dataset[:train_size]
    data_test = dataset[train_size:]
    print(len(data_train),len(data_test))

    #Init labels
    test_labels = []  
    for data in data_test:
        y = data.y
        test_labels.append(y.tolist())
    test_labels = np.array(test_labels)

    edge_save_path_1 = ("./edge_save/" + args.dataset + "_"+ str(args.initial_epochs)+ "_" 
                        + str(args.iter_num) + "_" + str(args.iter_epochs) + "_pos.pt")
    edge_save_path_2 = ("./edge_save/" + args.dataset + "_"+ str(args.initial_epochs)+ "_"
                        + str(args.iter_num) + "_" + str(args.iter_epochs) + "_not_neg.pt")
    pos_edges, not_neg_edges = [], []

    if os.path.exists(edge_save_path_1) == False or os.path.exists(edge_save_path_2) == False:
        print("Now begin initial training!")
        model, optimizer = initialize()
        model.train()
        for epoch in tqdm(range(args.initial_epochs)):
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                bce_loss, gen_loss, con_loss, h_t, _ = model(data_train, 0)
                loss = bce_loss 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
        print("Now begin iter training!")
        for iter in range(args.iter_num):
            print("current: ", iter)
            model.eval()
            if iter==0:
                pos_edges, not_neg_edges = model(data_train, 0)
            else:
                pos_edges, not_neg_edges = model(data_train, 0, pos_edges, not_neg_edges)
            if iter == args.iter_num - 1:
                break
            model, optimizer = initialize()
            model.train()
            for epoch in tqdm(range(args.iter_epochs)):
                optimizer.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    bce_loss, gen_loss, con_loss, h_t, _ = model(data_train, 1, pos_edges, not_neg_edges)
                    loss = bce_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
        if args.iter_num > 0:
            torch.save(pos_edges, edge_save_path_1)
            torch.save(not_neg_edges, edge_save_path_2)
    else:
        pos_edges = torch.load(edge_save_path_1)
        not_neg_edges = torch.load(edge_save_path_2)


    #Start Training
    print("Now begin last training!")
    model, optimizer = initialize()
    load_time = time.time()
    print(f'\n Total Loading Rime:{(load_time-start_time):.4f}')
    # model.train() 
    max_auc = 0.0   
    max_epoch = 0
    for epoch in tqdm(range(args.epochs)):
        model.train()
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            bce_loss, gen_loss, con_loss, h_t, _ = model(data_train, 2, pos_edges, not_neg_edges) 
            loss = bce_loss + 1.0 * con_loss + 1.0 * gen_loss  
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        if (epoch+1) % 5 == 0 or epoch == args.epochs-1:
            model.eval()
            with torch.no_grad():
                _, _, _, _, score_list = model(data_test, 3, h_t = h_t)
            score_all = []
            label_all = []
            for t in range(len(score_list)):
                score = score_list[t].cpu().numpy().squeeze()
                score_all.append(score)
                label_all.append(test_labels[t])
                auc = roc_auc_score(test_labels[t], score)
            score_all = np.hstack(score_all)
            label_all = np.hstack(label_all)
            auc_all = roc_auc_score(label_all, score_all)
            if max_auc <= auc_all:
                max_auc = auc_all
                max_epoch = epoch
            tb.add_scalar('auc_all', auc_all.item(), epoch)
            print(f"overall AUC: {auc_all:.4f}")

        tb.add_scalar('bce_loss', bce_loss.item(), epoch)
        tb.add_scalar('gen_loss', gen_loss.item(), epoch)
        tb.add_scalar('con_loss', con_loss.item(), epoch)
        tb.add_scalar('loss', loss.item(), epoch)
        print(f'bce_loss:{bce_loss.item():.4f} + gen_loss: {gen_loss.item():.4f} + con_loss: {con_loss.item():.4f}')
        print(f'train_loss: {loss.item():.4f}')
    tb.close()
    log.writelines("loss\tbce_loss\tgen_loss\tcon_loss\t\n")
    log.writelines(f'{loss:.3f}\t{bce_loss:.3f}\t{gen_loss:.3f}\t{con_loss:.3f}\t\n')
    log.writelines(f"MAX AUC: {max_auc:.4f} in epoch: {max_epoch}\n")
    print(f'\n Total Training Rime:{(time.time()-load_time):.4f}')
    torch.save(model.state_dict(), model_file)
    print(f"MAX AUC: {max_auc:.4f} in epoch: {max_epoch}")
    