#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   options.py
@Time    :   2021/07/19 10:49:40
@Author  :   Guo Jianhao 
'''

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='none', help='name of dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
    parser.add_argument('--epochs', type=int, default=10, help='training epochs')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')

    
    args = parser.parse_args()
    return args
