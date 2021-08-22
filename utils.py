#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/08/19 12:33:05
@Author  :   Guo Jianhao 
'''

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes','y','true','t','1'):
        return True
    if v.lower() in ('no','n','false','f','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    parser.add_argument('--dataset', type=str, default='None', help='name of dataset')
    parser.add_argument('--multi_gpu', type=str2bool, default=False, help='multi-gpu mode')
    parser.add_argument('--epochs', type=int, default=500, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--num_layers', type=int, default=2, help='gnn layers')
    parser.add_argument('--num_nodes', type=int, default=1809, help='number of all nodes')
    parser.add_argument('--in_channels_gnn', type=int, default=1809, help='in channels')
    parser.add_argument('--hidden_channels_gnn', type=int, default=512, help='hidden_channels')
    parser.add_argument('--out_channels_gnn', type=int, default=256, help='out_channels')
    # parser.add_argument('--out_channels_rnn', type=int, default=1809, help='out_channels')
    parser.add_argument('--add_ev', type=str2bool, default=True, help='add_ev')
    parser.add_argument('--n_samples', type=int, default=8, help='number of samples for contrastive learning')
    parser.add_argument('--timestamp', type=int, default=12, help='k timestamp for contrastive learning')
    parser.add_argument('--num_classes', type=int, default=2, help='normal or abnormal')
    args = parser.parse_args()
    return args

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset     : {args.dataset}')
    print(f'    Device      : {args.device}')
    print(f'    Multi-GPU   : {args.multi_gpu}')
    # print(f'    Model     : {args.model}')
    # print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning Rate   : {args.lr}')
    print(f'    Training Epochs : {args.epochs}')

    print('\n Network details:')
    print(f'    1st-GNN in: {args.in_channels_gnn}, out:{args.hidden_channels_gnn}')
    print(f'    2nd-GNN in: {args.hidden_channels_gnn}, out:{args.out_channels_gnn}')
    print(f'    Eigenvector as representation: {args.add_ev}')
    if args.add_ev:
        print(f'    RNN in {args.out_channels_gnn + args.num_nodes}, out:{args.out_channels_gnn + args.num_nodes}')
    else:
        print(f'    RNN in {args.out_channels_gnn}, out:{args.out_channels_gnn}')
    print(f'    N_samples:{args.n_samples}')
    print(f'    Timestamp:{args.timestamp}\n')

    return
