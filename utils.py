#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
# import networkx as nx
# import matplotlib.pyplot as plt  

# #visualize graphs
# for i,graph in enumerate(data):
#     draw(graph.edge_index, graph.y, i)

# def draw(edge_index, y, name=None):
#     G = nx.MultiGraph(node_size=15, font_size=8)
#     src = edge_index[0].cpu().numpy()
#     dst = edge_index[1].cpu().numpy()
#     edgelist = zip(src, dst)
#     for i, j in edgelist:
#         G.add_edge(i, j)
#     plt.figure(figsize=(20, 14)) # 设置画布的大小
#     if y == 1:
#         nx.draw_networkx(G,node_color="red")
#     else:
#         nx.draw_networkx(G,node_color="blue")
#     if not os.path.exists('figs'):
#         os.mkdir('figs')
#     plt.savefig('figs/{}.png'.format(name if name else 'path'))
#     print(f'Saved fig-{name}.')
    
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

    # ===== dataset parameters =====
    parser.add_argument('--dataset', type=str, default='uci', help='name of dataset')
    parser.add_argument('--snap_size', type=int, default=500, help='')
    parser.add_argument('--train_ratio', type=float, default=0.5, help='')
    parser.add_argument('--anomaly_ratio', type=float, default=0.1, help='')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='noise ratio')

    # ===== training parameters =====
    parser.add_argument('--epochs', type=int, default=250, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')

    # ===== hyper-parameters =====
    parser.add_argument('--window', type=int, default=1, help='')
    parser.add_argument('--eps', type=float, default=0.2, help='eps')
    parser.add_argument('--bce_weight', type=float, default=1, help='')
    parser.add_argument('--gen_weight', type=float, default=1, help='')
    parser.add_argument('--con_weight', type=float, default=1, help='')
    parser.add_argument('--reg_weight', type=float, default=1, help='')

    # ===== model parameters =====
    parser.add_argument('--layer_num', type=int, default=2, help='rnn layers')
    parser.add_argument('--x_dim', type=int, default=256, help='input channels of the model')
    parser.add_argument('--h_dim', type=int, default=256, help='hidden channels of the model')
    parser.add_argument('--z_dim', type=int, default=256, help='output channels of the model')

    # ===== may remove in the future =====
     # parser.add_argument('--hidden_dim_rnn', type=int, default=256, help='hidden channels of rnn')
    # parser.add_argument('--num_nodes', type=int, default=1809, help='number of all nodes')
    # parser.add_argument('--num_layers_gae', type=int, default=2, help='gnn layers')
    # parser.add_argument('--input_dim_gae', type=int, default=1809, help='input channels of gae')
    # parser.add_argument('--hidden_dim_gae', type=int, default=512, help='hidden channels of gae')
    # parser.add_argument('--add_ev', type=str2bool, default=True, help='add_ev')
    # parser.add_argument('--timespan', type=int, default=6, help='k timestamp for contrastive learning')
    # parser.add_argument('--num_classes', type=int, default=2, help='normal or abnormal')
    args = parser.parse_args()
    return args

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset     : {args.dataset}')
    print(f'    Device      : {args.device}')
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
