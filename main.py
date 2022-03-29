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
from model import LR_Scheduler, Model
from dataset import  DynamicGraphAnomaly
from negativesampler import NegativeSampler

def mse(mat1, mat2, axis=0):
    assert mat1.size() == mat2.size()
    loss = []
    for i in range(mat1.size(0)):
        loss.append(nn.MSELoss(reduction='sum')(mat1[i], mat2[i]).item())
    return loss

def evaluate(labels, anomaly_scores, log, threshold = None):
    len = anomaly_scores.size()[0]
    new_anomaly_scores, _ = torch.sort(anomaly_scores, descending=True)
    for i, label in enumerate(labels):
        if label == 1:
            anomaly_score = anomaly_scores[i]
            index1, index2 = len, len
            flag1, flag2 = 0, 0
            for j, score in enumerate(new_anomaly_scores):
                if score <= anomaly_score and flag1 == 0:
                    index1 = j
                    flag1 = 1
                if score < anomaly_score and flag2 == 0:
                    index2 = j
                    flag2 = 1
                    break
            # log.writelines(f"The {i} th graph is an anomaly graph whose anomaly score is: {anomaly_score}.\n")
            # log.writelines(f"There are {index1} graphs with a higher anomaly score.\n")
            # log.writelines(f"There are {len - index2} graphs with a lower anomaly score.\n")
            print(f"The {i} th graph is an anomaly graph whose anomaly score is: {anomaly_score}.\n")
            print(f"There are {index1} graphs with a higher anomaly score.\n")
            print(f"There are {len - index2} graphs with a lower anomaly score.\n")
    if threshold != None:
        y_true = labels.detach().numpy()
        y_score = anomaly_scores.detach().numpy()
        y_score1 = [1 if i > threshold else 0 for i in anomaly_scores]
        auc = roc_auc_score(y_true, y_score)
        auc1 = roc_auc_score(y_true, y_score1)
        return auc, auc1
    
def draw_graphs(dataset, training = True):
    if training == True:
        name = 'train_'
    else:
        name = 'test_'
    for t, graph in enumerate(dataset):
        edge_index = graph.edge_index
        y = graph['y'].item()
        draw(edge_index, y, name+str(t))

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
    tb_path = f'runs/{args.dataset}/'
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    tb = SummaryWriter(logdir=tb_path+f"{time.strftime('%m-%d,%H:%M:%S')}_snaps{args.snap_size}_train{args.train_ratio}_anomaly{args.anomaly_ratio}_epochs{args.epochs}_lr{args.lr}_xdim{args.x_dim}_hzdim{args.z_dim}_eps{args.eps}_lambda{args.lamda}")
    #log
    log_path = f'logs/{args.dataset}/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, f'snaps{args.snap_size}_train{args.train_ratio}_anomaly{args.anomaly_ratio}_epochs{args.epochs}_lr{args.lr}_xdim{args.x_dim}_hzdim{args.z_dim}_eps{args.eps}_lambda{args.lamda}')
    log = open(log_file, "a")
    log.writelines(time.strftime('%m-%d %H:%M:%S') + "\n")
    log.writelines(str(args) + "\n")
    #model
    model_path = f'./models/{args.dataset}/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_file = os.path.join(model_path, f'snaps{args.snap_size}_train{args.train_ratio}_anomaly{args.anomaly_ratio}_epochs{args.epochs}_lr{args.lr}_xdim{args.x_dim}_hzdim{args.z_dim}_eps{args.eps}_lambda{args.lamda}.pkl')

    # exp_details(args)
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
    #if True:
        #Init model
        # model = Model(args).to(args.device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print("Now begin initial training!")
        model, optimizer = initialize()
        model.train()
        for epoch in tqdm(range(args.initial_epochs)):
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                bce_loss, recon_loss, kld_loss, nce_loss, h_t, _ = model(data_train, 0)
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
                    bce_loss, recon_loss, kld_loss, nce_loss, h_t, _ = model(data_train, 1, pos_edges, not_neg_edges)
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
    print("Now begin last training!")
    model, optimizer = initialize()
    # lr_scheduler = LR_Scheduler(optimizer,args.warmup_epochs, args.warmup_lr/256, args.epochs, args.lr/256, args.lr/256, len(data_train), )
    load_time = time.time()
    print(f'\n Total Loading Rime:{(load_time-start_time):.4f}')

    #Start Training
    
    # model.train() 
    max_auc = 0.0   
    max_epoch = 0
    for epoch in tqdm(range(args.epochs)):
        model.train()
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            bce_loss, recon_loss, kld_loss, nce_loss, h_t, _ = model(data_train, 2, pos_edges, not_neg_edges)
            # if epoch < 50:
            #     loss = nce_loss + recon_loss + kld_loss #+ bce_loss 
            # else:
            #     loss = bce_loss    
            loss = bce_loss + 1.0 * nce_loss + 1.0 * (recon_loss + kld_loss)  
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        # lr_scheduler.step()

        if (epoch+1) % 5 == 0 or epoch == args.epochs-1:
            model.eval()
            with torch.no_grad():
                _, _, _, _, _, score_list = model(data_test, 3, h_t = h_t)
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
        tb.add_scalar('recon_loss', recon_loss.item(), epoch)
        tb.add_scalar('kld_loss', kld_loss.item(), epoch)
        tb.add_scalar('jsd_loss', nce_loss.item(), epoch)
        #tb.add_scalar('regularizer', regularizer.item(), epoch)
        tb.add_scalar('loss', loss.item(), epoch)
        print(f'bce_loss:{bce_loss.item():.4f} + recon_loss: {recon_loss.item():.4f} + kld_loss: {kld_loss.item():.4f} + nce_loss: {nce_loss.item():.4f}')
        print(f'train_loss: {loss.item():.4f}')
    tb.close()
    log.writelines("loss\tbce_loss\trecon_loss\tkld_loss\tjsd_loss\tregularizer\t\n")
    log.writelines(f'{loss:.3f}\t{bce_loss:.3f}\t{recon_loss:.3f}\t{kld_loss:.3f}\t{nce_loss:.3f}\t\n')
    print(f'\n Total Training Rime:{(time.time()-load_time):.4f}')
    torch.save(model.state_dict(), model_file)
    
    print(f"MAX AUC: {max_auc:.4f} in epoch: {max_epoch}")
    