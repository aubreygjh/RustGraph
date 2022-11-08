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

def update_reduce_step(cur_step, num_gradual, tau=0.5):
    return 1.0 - tau * min(cur_step / num_gradual, 1)

def co_teaching_loss(model1_loss, model2_loss, rt):
    _, model1_sm_idx = torch.topk(model1_loss, k=int(int(model1_loss.size(0)) * rt), largest=False)
    _, model2_sm_idx = torch.topk(model2_loss, k=int(int(model2_loss.size(0)) * rt), largest=False)

    # co-teaching
    model1_loss_filter = torch.zeros((model1_loss.size(0))).cuda()
    model1_loss_filter[model2_sm_idx] = 1.0
    model1_loss = (model1_loss_filter * model1_loss).mean()

    model2_loss_filter = torch.zeros((model2_loss.size(0))).cuda()
    model2_loss_filter[model1_sm_idx] = 1.0
    model2_loss = (model2_loss_filter * model2_loss).mean()

    return model1_loss, model2_loss

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
    tb = SummaryWriter(logdir=tb_path+f"{time.strftime('%m-%d,%H:%M:%S')}")

    #log
    log_path = f'logs/{args.dataset}/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, f'{args.anomaly_ratio}')
    log = open(log_file, "a")
    log.writelines(time.strftime('%m-%d %H:%M:%S') + "\n")
    # log.writelines(str(args) + "\n")

    #model
    # model_path = f'models/{args.dataset}/'
    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)
    # model_file = os.path.join(model_path, f'{args.anomaly_ratio}.pkl')

    #Init dataloader
    dataset = []
    if args.dataset not in ['uci', 'digg', 'btc_alpha', 'btc_otc', 'email', 'as_topology']:
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

    #Start Training
    print("Now begin training!")
    model1, optimizer1 = initialize()
    model2, optimizer2 = initialize()

    load_time = time.time()
    print(f'\n Total Loading Rime:{(load_time-start_time):.4f}')
    max_auc = 0.0   
    max_epoch = 0

    model1.train()
    model2.train()
    y_new1 = None
    y_new2 = None
    for epoch in tqdm(range(args.epochs)):
        rt = update_reduce_step(cur_step=epoch, num_gradual=15)
        with torch.autograd.set_detect_anomaly(True):
            bce_loss1, reg_loss1, gen_loss1, con_loss1, y_new1, h_t1, _ = model1(data_train, y_rect=y_new1)
            # bce_loss2, reg_loss2, gen_loss2, con_loss2, y_new2, h_t2, _ = model2(data_train, y_rect=y_new1)
            
            loss1 = bce_loss1.mean()
            # loss1, loss2 = co_teaching_loss(bce_loss1, bce_loss2, rt=rt)
            loss1 += args.reg_weight * reg_loss1 + args.gen_weight * gen_loss1 + args.con_weight * con_loss1
            # loss2 += args.reg_weight * reg_loss2 + args.gen_weight * gen_loss2 + args.con_weight * con_loss2
            optimizer1.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(model1.parameters(), 10)
            optimizer1.step()

            # optimizer2.zero_grad()
            # loss2.backward()
            # torch.nn.utils.clip_grad_norm_(model2.parameters(), 10)
            # optimizer2.step()

        if (epoch+1) % 5 == 0 or epoch == args.epochs-1:
            model1.eval()
            with torch.no_grad():
                _, _, _, _, _,_, score_list = model1(data_test, h_t=h_t1)
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

        tb.add_scalar('bce_loss', bce_loss1.mean().item(), epoch)
        tb.add_scalar('reg_loss', reg_loss1.item(), epoch)
        tb.add_scalar('gen_loss', gen_loss1.item(), epoch)
        tb.add_scalar('con_loss', con_loss1.item(), epoch)
        tb.add_scalar('loss', loss1.item(), epoch)

    tb.close()
    log.writelines(f"MAX AUC: {max_auc:.4f} in epoch: {max_epoch},\t")
    log.writelines(f"Last AUC: {auc_all:.4f}\n")
    print(f'\n Total Training Rime:{(time.time()-load_time):.4f}')
    # torch.save(model1.state_dict(), model_file)
    print(f"MAX AUC: {max_auc:.4f} in epoch: {max_epoch}")
    