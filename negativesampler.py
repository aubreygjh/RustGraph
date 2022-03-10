import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, InnerProductDecoder
from torch_geometric.utils import negative_sampling
class GConv(nn.Module):
    def __init__(self, input_dim, output_dim, device, act = None, bias = True, dropout = 0.):
        super(GConv, self).__init__()
        # self.conv = SAGEConv(input_dim, output_dim).to(device)

        self.conv = GATConv(input_dim, output_dim, edge_dim=64).to(device)
        self.dropout = dropout
        self.act = act
    
    def forward(self, x, edge_index):
        z = self.conv(x, edge_index)
        if self.act != None:
            z = self.act(z)
        z = F.dropout(z, p = self.dropout, training = self.training)
        return z 

class NegativeSampler(nn.Module):
    def __init__(self, args):
        super(NegativeSampler, self).__init__()
        self.gnn = GConv(args.x_dim, args.z_dim, args.device)
        self.dec = InnerProductDecoder()
        self.device = args.device
    def forward(self, dataloader):
        neg_edge_index_list = []
        for data in dataloader:
            data = data.to(self.device)
            x = data.x
            edge_index = data.edge_index
            node_index = data.node_index

            # neg_edge_index = negative_sampling(edge_index)
            z = self.gnn(x, edge_index)
            dense_adj = self.dec.forward_all(z)
            node_num = dense_adj.size(0)
            edge_num = edge_index.size(1)

            dense_adj_flatten = dense_adj.view(-1)

            #这部分可以bp吗
            val, idx = torch.topk(dense_adj_flatten, edge_num)

            src = (idx / node_num).view(1, -1).long()
            dst = (idx % node_num).view(1, -1).long()

            neg_edge_index = torch.cat([src, dst], dim=0)

            # neg_edge_index = copy.deepcopy(edge_index)
            neg_edge_index_list.append(neg_edge_index)
        return neg_edge_index_list