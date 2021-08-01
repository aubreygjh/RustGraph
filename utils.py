#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2021/07/16 10:50:01
@Author  :   Guo Jianhao 
'''


import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='none', help='name of dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--num_layers', type=int, default=3, help='gnn layers')
    parser.add_argument('--in_channels', type=int, default=3, help='in channels')
    parser.add_argument('--hidden_channels', type=int, default=6, help='')
    parser.add_argument('--out_channels', type=int, default=3, help='')
    parser.add_argument('--num_classes', type=int, default=6, help='')
    args = parser.parse_args()
    return args

def exp_details(args):
    print('details')
    return