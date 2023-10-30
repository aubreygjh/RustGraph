#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from utils import *
from model import  Model
from ae_model import AE_Model
from dataset import  DynamicGraphAnomaly
'''
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
'''
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
    tb = SummaryWriter(log_dir=tb_path+f"{time.strftime('%m-%d,%H:%M:%S')}")

    #log
    log_path = f'logs/{args.dataset}/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, f'{args.anomaly_ratio}')
    log = open(log_file, "a")
    log.writelines(time.strftime('%m-%d %H:%M:%S') + "\n")

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
    # test_labels = np.array(test_labels)

    #Start Training
    print("Now begin training!")
    model, optimizer = initialize()

    load_time = time.time()
    print(f'\n Total Loading Rime:{(load_time-start_time):.4f}')
    max_auc = 0.0   
    max_epoch = 0

    model.train()
    y_new = None
    for epoch in tqdm(range(args.epochs)):
        with torch.autograd.set_detect_anomaly(True):
            bce_loss, reg_loss, gen_loss, con_loss, y_new, h_t, _ = model(data_train, y_rect=y_new)
            loss = bce_loss.mean()
            loss = args.bce_weight * loss + args.reg_weight * reg_loss + args.gen_weight * gen_loss + args.con_weight * con_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        if (epoch+1) % 5 == 0 or epoch == args.epochs-1:
            model.eval()
            with torch.no_grad():
                _, _, _, _, _,_, score_list = model(data_test, h_t=h_t)
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
            # ###train_ratio exp
            # if epoch == args.epochs-1:
            #     for t in range(len(score_list)):
            #         score = score_list[t].cpu().numpy().squeeze()
            #         auc = roc_auc_score(test_labels[t], score)
            #         log.writelines(f"{auc:.4f}, ")
            #     log.writelines("\n")
            
            tb.add_scalar('auc_all', auc_all.item(), epoch)
            # print(f"overall AUC: {auc_all:.4f}")

        tb.add_scalar('bce_loss', bce_loss.mean().item(), epoch)
        tb.add_scalar('reg_loss', reg_loss.item(), epoch)
        tb.add_scalar('gen_loss', gen_loss.item(), epoch)
        tb.add_scalar('con_loss', con_loss.item(), epoch)
        # tb.add_scalar('KL divergence', kld_loss.item(), epoch)
        # tb.add_scalar('Reconstruction loss', recon_loss.item(), epoch)
        tb.add_scalar('loss', loss.item(), epoch)

    tb.close()
    log.writelines(f"MAX AUC: {max_auc:.4f} in epoch: {max_epoch},\t")
    log.writelines(f"Last AUC: {auc_all:.4f}\n")
    print(f'\n Total Training Rime:{(time.time()-load_time):.4f}')
    # torch.save(model.state_dict(), model_file)
    print(f"MAX AUC: {max_auc:.4f} in epoch: {max_epoch}")
    