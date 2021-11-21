import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import SAGEConv

class GConv(nn.Module):
    def __init__(self, input_dim, output_dim, device, act = None, bias = True, dropout = 0.):
        super(GConv, self).__init__()
        self.conv = SAGEConv(input_dim, output_dim).to(device)
        self.dropout = dropout
        self.act = act
    
    def forward(self, x, edge_index):
        z = self.conv(x, edge_index)
        if self.act != None:
            z = self.act(z)
        z = F.dropout(z, p = self.dropout, training = self.training)
        return z 


class Graph_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num, device, bias = True):
        super(Graph_GRU, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []
        
        for i in range(layer_num):
            if i == 0:
                self.weight_xz.append(GConv(input_size, hidden_size, device=device, bias=bias)) 
                self.weight_hz.append(GConv(hidden_size, hidden_size, device=device, bias=bias)) 
                self.weight_xr.append(GConv(input_size, hidden_size, device=device, bias=bias)) 
                self.weight_hr.append(GConv(hidden_size, hidden_size, device=device, bias=bias)) 
                self.weight_xh.append(GConv(input_size, hidden_size, device=device, bias=bias)) 
                self.weight_hh.append(GConv(hidden_size, hidden_size, device=device, bias=bias)) 
            else:
                self.weight_xz.append(GConv(hidden_size, hidden_size, device=device, bias=bias)) 
                self.weight_hz.append(GConv(hidden_size, hidden_size, device=device, bias=bias)) 
                self.weight_xr.append(GConv(hidden_size, hidden_size, device=device, bias=bias)) 
                self.weight_hr.append(GConv(hidden_size, hidden_size, device=device, bias=bias)) 
                self.weight_xh.append(GConv(hidden_size, hidden_size, device=device, bias=bias)) 
                self.weight_hh.append(GConv(hidden_size, hidden_size, device=device, bias=bias)) 
    
    def forward(self, x, edge_index, h):
        h_out = torch.zeros(h.size()).to(self.device)
        for i in range(self.layer_num):
            if i == 0:
                z_g = torch.sigmoid(self.weight_xz[i](x, edge_index) + self.weight_hz[i](h[i], edge_index))
                r_g = torch.sigmoid(self.weight_xr[i](x, edge_index) + self.weight_hr[i](h[i], edge_index))
                h_tilde_g = torch.tanh(self.weight_xh[i](x, edge_index) + self.weight_hh[i](r_g * h[i], edge_index))
                out = z_g * h[i] + (1 - z_g) * h_tilde_g
            else:
                z_g = torch.sigmoid(self.weight_xz[i](out, edge_index) + self.weight_hz[i](h[i], edge_index))
                r_g = torch.sigmoid(self.weight_xr[i](out, edge_index) + self.weight_hr[i](h[i], edge_index))
                h_tilde_g = torch.tanh(self.weight_xh[i](out, edge_index) + self.weight_hh[i](r_g * h[i], edge_index))
                out = z_g * h[i] + (1 - z_g) * h_tilde_g
            h_out[i] = out
        return h_out



class Generative(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, layer_num, device):
        super(Generative, self).__init__()
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())
        
        self.enc = GConv(h_dim + h_dim, h_dim, device=device, act=F.relu)
        self.enc_mean = GConv(h_dim, z_dim, device=device)
        self.enc_std = GConv(h_dim, z_dim, device=device, act=F.softplus)
        
        self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
        self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
        self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())
        self.device = device

        self.rnn = Graph_GRU(h_dim+h_dim, h_dim, layer_num, device)

    
    def forward(self, x, h, edge_index):
        phiX = self.phi_x(x)
        enc_x = self.enc(torch.cat([phiX, h[-1]], 1), edge_index)
        enc_x_mean = self.enc_mean(enc_x, edge_index)
        enc_x_std = self.enc_std(enc_x, edge_index)
        prior_x = self.prior(h[-1])
        prior_x_mean = self.prior_mean(prior_x)
        prior_x_std = self.prior_std(prior_x)
        z = self.random_sample(enc_x_mean, enc_x_std)
        phiZ = self.phi_z(z)
        h_out = self.rnn(torch.cat([phiX, phiZ], 1), edge_index, h)
        
        return (prior_x_mean, prior_x_std), (enc_x_mean, enc_x_std), z, h_out
    
    def random_sample(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps1 = Variable(eps1)
        return eps1.mul(std).add_(mean)


class InnerProductDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid, dropout=0.):
        super(InnerProductDecoder, self).__init__()
        self.act = act
        self.dropout = dropout
    
    def forward(self, inp):
        inp = F.dropout(inp, self.dropout, training=self.training)
        x = torch.transpose(inp, dim0=0, dim1=1)
        x = torch.mm(inp, x)
        return self.act(x)


class CPC(nn.Module):
    def __init__(self, device, sample_num, timespan, h_dim, z_dim):
        super(CPC, self).__init__()
        self.device = device
        self.sample_num = sample_num
        self.timespan = timespan
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.h_project = nn.Linear(h_dim, h_dim)
        self.z_project = nn.Linear(z_dim, h_dim)
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, all_h, all_z):
        h_pooling = self.graph_pooling(all_h, self.h_dim, self.h_dim)
        z_pooling = self.graph_pooling(all_z, self.z_dim, self.h_dim)
        seq_len = z_pooling.shape[0]
        feat_dim = z_pooling.shape[1]
        neg_dist = int(seq_len/6)
        end = seq_len - self.sample_num - neg_dist - self.timespan + 2
        start = int(seq_len/8) if int(seq_len/8) < end else 0
        cnt = 0
        nce_loss = 0
        distance = 0
        # correct = 0
        global_mean = torch.mean(z_pooling, dim=0)
        for i in range(seq_len):
            #print(z_pooling[i].size())
            #print(global_mean.size())
            diff = torch.abs(z_pooling[i] - global_mean)
            distance += torch.sum(diff*diff)

            #distance += F.pairwise_distance(z_pooling[i], global_mean)
        for t_sample in range(start, end):
            cnt += 1
            encode_samples = torch.empty((self.timespan, self.sample_num, feat_dim)).float().to(self.device)
            for i in np.arange(1, self.timespan+1):
                encode_samples[i-1][0] = z_pooling[t_sample+i,:].view(feat_dim)
                for n_sample in np.arange(1, self.sample_num):
                    encode_samples[i-1][n_sample] = z_pooling[t_sample+i+neg_dist+n_sample-1,:].view(feat_dim)
            c_t = h_pooling[t_sample,:].view(feat_dim)

            for i in np.arange(0, self.timespan):
                c_phi_t = nn.Linear(feat_dim, feat_dim)(c_t)
                total = F.cosine_similarity(encode_samples[i], c_phi_t, dim=-1)
                nce_loss += self.lsoftmax(total)[0]
        nce_loss /= -1.*cnt*self.timespan
        distance /= 1.*seq_len

        return nce_loss, distance        
    
    def graph_pooling(self, x, x_dim, z_dim):
        x_out = torch.empty((len(x), z_dim)).float().to(self.device)
        for i, x_t in enumerate(x):
            x_t = torch.mean(x_t, dim = 0)
            if x_dim == self.h_dim:
                x_t = self.h_project(x_t)
            else:
                x_t = self.z_project(x_t)
            x_out[i] = x_t
        return x_out


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.conv = Generative(args.x_dim, args.h_dim, args.z_dim, args.layer_num, args.device)
        self.dec = InnerProductDecoder()
        self.eps = args.eps
        self.device = args.device
        self.layer_num = args.layer_num
        self.h_dim = args.h_dim
        self.lamda = args.lamda
        self.cpc = CPC(args.device, args.sample_num, args.timespan, args.h_dim, args.z_dim)
    
    def forward(self, dataloader, all_time_nodes, all_time_adj):
        kld_loss = 0
        nll_loss = 0
        all_enc_mean, all_enc_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_dec, all_z, all_h = [], [], []
        h_t = Variable(None)
        # print(dataloader.__len__())
        for t, data in enumerate(dataloader):
            data = data.to(self.device)
            x = data.x
            edge_index = data.edge_index
            nodes = all_time_nodes[t]
            adj = all_time_adj[t]
            if t == 0:
                h_t = Variable(torch.zeros(self.layer_num, x.size(0), self.h_dim).to(self.device))
            
            (prior_mean_t, prior_std_t), (enc_mean_t, enc_std_t), z_t, h_t = self.conv(x, h_t, edge_index)
            print('mid: ', t)
            dec_t = self.dec(z_t)
            enc_mean_t_sl = enc_mean_t[nodes, :]
            enc_std_t_sl = enc_std_t[nodes, :]
            prior_mean_t_sl = prior_mean_t[nodes, :]
            prior_std_t_sl = prior_std_t[nodes, :]
            dec_t_sl = dec_t[nodes, :][:,nodes]
            z_t_sl = z_t[nodes, :]
            h_t_sl = h_t[-1, nodes, :]

            kld_loss += self._kld_gauss(enc_mean_t_sl, enc_std_t_sl, prior_mean_t_sl, prior_std_t_sl)
            nll_loss += self._nll_bernoulli(dec_t_sl, adj)
            all_enc_mean.append(enc_mean_t_sl)
            all_enc_std.append(enc_std_t_sl)
            all_prior_mean.append(prior_mean_t_sl)
            all_prior_std.append(prior_std_t_sl)
            all_dec.append(dec_t_sl)
            all_z.append(z_t_sl)
            all_h.append(h_t_sl)
        
        nce_loss, distance = self.cpc(all_z, all_h)
        return kld_loss + nll_loss + self.lamda * (nce_loss + self.eps * distance)

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)
     
    def _init_weights(self, stdv):
        pass

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element =  (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mean_1 - mean_2, 2)) / 
                        torch.pow(std_2 + self.eps ,2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)
    
    def _kld_gauss_zu(self, mean_in, std_in):
        num_nodes = mean_in.size()[0]
        std_log = torch.log(std_in + self.eps)
        kld_element =  torch.mean(torch.sum(1 + 2 * std_log - mean_in.pow(2) -
                                            torch.pow(torch.exp(std_log), 2), 1))
        return (-0.5 / num_nodes) * kld_element

    def _nll_bernoulli(self, logits, target_adj_dense):
        temp_size = target_adj_dense.size()[0]
        temp_sum = target_adj_dense.sum()
        posw = float(temp_size * temp_size - temp_sum) / temp_sum
        norm = temp_size * temp_size / float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(input=logits
                                                          , target=target_adj_dense
                                                          , pos_weight=posw
                                                          , reduction='none')
        nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0,1])
        return - nll_loss

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels // 16),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(in_channels // 16, out_channels)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        return x, self.softmax(x)


