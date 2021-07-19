#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/07/19 17:36:57
@Author  :   Guo Jianhao 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv

class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel,self).__init__()
        self.args = args

    def forward(self, data):
        x = ...
        return
    