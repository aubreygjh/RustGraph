import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as data
from torch_geometric.nn import global_max_pool 
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import numpy as np
from scipy.sparse.linalg import eigs, eigsh

def compute_ev(data,normalization=None, is_undirected=False):
    assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'
    edge_weight = data.edge_attr
    if edge_weight is not None and edge_weight.numel() != data.num_edges:
        edge_weight = None
    edge_index, edge_weight = get_laplacian(data.edge_index, edge_weight,
                                            normalization,
                                            num_nodes=data.num_nodes)
    L = to_scipy_sparse_matrix(edge_index, edge_weight, data.num_nodes)
    eig_fn = eigs
    if is_undirected and normalization != 'rw':
        eig_fn = eigsh
    lambda_max,ev = eig_fn(L, k=1, which='LM', return_eigenvectors=True)
    return ev #eigen_vectors of shape(dim, k); dim=length of matrix, k=top-k vectors

e1 = torch.tensor([[0,0,1,2],[1,2,0,0]])
print(e1.shape)
x1 = torch.tensor([[1],[1]])
x2 = torch.tensor([[1],[1],[1]])
x3 = torch.tensor([[1],[1],[1],[1]])
print(e1.shape)
g1 = data.Data(x = x3,edge_index=e1)
ev = compute_ev(g1)
print(ev)


