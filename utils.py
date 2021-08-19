#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/08/19 12:33:05
@Author  :   Guo Jianhao 
'''

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='None', help='name of dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
    parser.add_argument('--epochs', type=int, default=500, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--num_layers', type=int, default=2, help='gnn layers')
    parser.add_argument('--in_channels', type=int, default=1809, help='in channels')
    parser.add_argument('--hidden_channels', type=int, default=1000, help='hidden_channels')
    parser.add_argument('--out_channels', type=int, default=1809, help='out_channels')
    parser.add_argument('--n_samples', type=int, default=8, help='number of samples for contrastive learning')
    parser.add_argument('--timestamp', type=int, default=12, help='k timestamp for contrastive learning')
    parser.add_argument('--num_classes', type=int, default=2, help='normal or abnormal')
    args = parser.parse_args()
    return args

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset     : {args.dataset}')
    print(f'    Device      : {args.device}')
    # print(f'    Model     : {args.model}')
    # print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning Rate   : {args.lr}')
    print(f'    Training Epochs : {args.epochs}')

    print('\n Network details:')
    print(f'    1st-GNN in: {args.in_channels}, out:{args.hidden_channels*2}')
    print(f'    2nd-GNN in: {args.hidden_channels*2}, out:{args.hidden_channels}')
    print(f'    RNN in {args.hidden_channels}, out:{args.out_channels}')
    print(f'    N_samples:{args.n_samples}')
    print(f'    Timestamp:{args.timestamp}\n')

    return
