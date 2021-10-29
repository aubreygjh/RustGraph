import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as data
from torch_geometric.nn import global_max_pool 
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import numpy as np
from scipy.sparse.linalg import eigs, eigsh
import networkx as nx
import matplotlib.pyplot as plt  
def draw(edge_index, y, name=None):
    G = nx.MultiGraph(node_size=15, font_size=8)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    edgelist = zip(src, dst)
    for i, j in edgelist:
        G.add_edge(i, j)
    plt.figure(figsize=(20, 14)) # 设置画布的大小
    if y == 1:
        nx.draw_networkx(G,node_color="red")
    else:
        nx.draw_networkx(G,node_color="blue")
    plt.savefig('figs/{}.png'.format(name if name else 'path'))
    print(f'Saved fig-{name}.')
