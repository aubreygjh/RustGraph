#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2021/07/16 10:50:01
@Author  :   Guo Jianhao 
'''


import argparse
from pickle import TRUE
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='none', help='name of dataset')
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
    parser.add_argument('--epochs', type=int, default=500, help='training epochs')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--num_layers', type=int, default=3, help='gnn layers')
    parser.add_argument('--in_channels', type=int, default=3, help='in channels')
    parser.add_argument('--hidden_channels', type=int, default=6, help='')
    parser.add_argument('--out_channels', type=int, default=6, help='')
    parser.add_argument('--num_classes', type=int, default=6, help='')
    args = parser.parse_args()
    return args

def exp_details(args):
    print('details')
    return

def from_networkx(G, x, group_node_attrs=None,group_edge_attrs = None):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or all, optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or all, optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.
    """
    import networkx as nx
    import torch_geometric

    G = nx.convert_node_labels_to_integers(G)
    # G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges())).t().contiguous()
    

    data = {}

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}
   
    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass
    
    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    
    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = [data[key] for key in group_node_attrs]
        xs = [x.view(-1, 1) if x.dim() <= 1 else x for x in xs]
        data.x = torch.cat(xs, dim=-1)
    ##Add feature matrix directly from parameter(Need to Modify Later)
    data.x = torch.from_numpy(x).float()

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        edge_attrs = [data[key] for key in group_edge_attrs]
        edge_attrs = [x.view(-1, 1) if x.dim() <= 1 else x for x in edge_attrs]
        data.edge_attr = torch.cat(edge_attrs, dim=-1)
    return data
