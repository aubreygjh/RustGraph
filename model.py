import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch_geometric.nn import SAGEConv,GATConv,InnerProductDecoder
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, get_laplacian, to_scipy_sparse_matrix, is_undirected
from scipy.sparse.linalg import eigs, eigsh


class GConv(nn.Module):
    def __init__(self, input_dim, output_dim, device, act = None, bias = True, dropout = 0.):
        super(GConv, self).__init__()
        self.conv = SAGEConv(input_dim, output_dim).to(device)
        #self.conv = GATConv(input_dim, output_dim, edge_dim=64).to(device)
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
        
        self.prior = nn.Sequential(nn.Linear(h_dim+1, h_dim), nn.ReLU())
        self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
        self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())
        self.device = device

        self.rnn = Graph_GRU(h_dim+h_dim, h_dim, layer_num, device)

    
    def forward(self, x, h, diff, edge_index):
        phiX = self.phi_x(x)
        enc_x = self.enc(torch.cat([phiX, h[-1]], 1), edge_index)
        enc_x_mean = self.enc_mean(enc_x, edge_index)
        enc_x_std = self.enc_std(enc_x, edge_index)
        prior_x = self.prior(torch.cat([h[-1], diff], 1))
        # prior_x = torch.randn(prior_x.shape).cuda()
        prior_x_mean = self.prior_mean(prior_x)
        prior_x_std = self.prior_std(prior_x)
        z = self.random_sample(enc_x_mean, enc_x_std)
        phiZ = self.phi_z(z)
        h_out = self.rnn(torch.cat([phiX, phiZ], 1), edge_index, h)
        
        return (prior_x_mean, prior_x_std), (enc_x_mean, enc_x_std), z, h_out
    
    def random_sample(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_().to(self.device)
        return eps1.mul(std).add_(mean)

class Contrastive(nn.Module):
    def __init__(self, device, z_dim, window):
        super(Contrastive, self).__init__()
        self.device = device
        self.max_dis = window
        self.linear = nn.Linear(z_dim, z_dim)
    
    def forward(self, all_z, all_node_idx):
        t_len = len(all_node_idx)
        nce_loss = 0
        f = lambda x: torch.exp(x)
        # self.neg_sample = last_h
        for i in range(t_len - self.max_dis):
            for j in range(i+1, i+self.max_dis+1):
                nodes_1, nodes_2 = all_node_idx[i].tolist(), all_node_idx[j].tolist()
                common_nodes = list(set(nodes_1) & set(nodes_2))
                z_anchor = all_z[i][common_nodes]
                z_anchor = self.linear(z_anchor)
                positive_samples = all_z[j][common_nodes]
                pos_sim = f(self.sim(z_anchor, positive_samples, True))
                neg_sim = f(self.sim(z_anchor, all_z[j], False))
                #index = torch.LongTensor(common_nodes).unsqueeze(1).to(self.device)
                neg_sim = neg_sim.sum(dim=-1).unsqueeze(1) #- torch.gather(neg_sim, 1, index)
                nce_loss += -torch.log(pos_sim / (neg_sim)).mean()
                # nce_loss += -(torch.log(pos_sim / (pos_sim + neg_sim.sum(dim=-1) - torch.gather(neg_sim, 1, index)))).mean()
        return nce_loss / (self.max_dis * (t_len - self.max_dis))   

    def sim(self, h1, h2, pos=False):
        z1 = F.normalize(h1, dim=-1, p=2)
        z2 = F.normalize(h2, dim=-1, p=2)
        if pos == True:
            return torch.einsum('ik, ik -> i', z1, z2).unsqueeze(1)
        else:
            return torch.mm(z1, z2.t())       
                

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.encoder = Generative(args.x_dim, args.h_dim, args.z_dim, args.layer_num, args.device)
        self.contrastive = Contrastive(args.device, args.z_dim, args.window)
        self.dec = InnerProductDecoder()
        self.mse = nn.MSELoss(reduction='mean')
        self.fcc = FCC(args.z_dim, 1, args.device)
        self.linear = nn.Sequential(nn.Linear(args.z_dim, args.x_dim), nn.ReLU())

        self.eps = args.eps
        self.device = args.device
        self.layer_num = args.layer_num
        self.h_dim = args.h_dim
        self.z_dim = args.z_dim
        self.EPS = 1e-15

    
    def forward(self, dataloader, y_rect=None, h_t=None):
        kld_loss = 0
        recon_loss = 0
        reg_loss = 0
        all_z, all_h, all_node_idx = [], [], []
        score_list = []
        next_y_list = []

        for t, data in enumerate(dataloader):
            data = data.to(self.device)
            x = data.x
            if y_rect == None:
                y = data.y.unsqueeze(1).float()
            else:
                y = y_rect[t]
            # y = data.y.unsqueeze(1).float()
            edge_index = data.edge_index
            node_index = data.node_index   
            if h_t == None:
                h_t = torch.zeros(self.layer_num, x.size(0), self.h_dim).to(self.device)
            ev=self._compute_ev(data, is_undirected=True)
            if t == 0:
                diff = torch.zeros(x.size(0), 1).to(self.device)
            else:
                diff = torch.abs(torch.sub(ev, pre_ev)).to(self.device)
            pre_ev = ev
            
            (prior_mean_t, prior_std_t), (enc_mean_t, enc_std_t), z_t, h_t = self.encoder(x, h_t, diff, edge_index)

            enc_mean_t_sl = enc_mean_t[node_index, :]
            enc_std_t_sl = enc_std_t[node_index, :]
            prior_mean_t_sl = prior_mean_t[node_index, :]
            prior_std_t_sl = prior_std_t[node_index, :]
            h_t_sl = h_t[-1, node_index, :]
            
            edge_emb = z_t[edge_index[0]] + z_t[edge_index[1]]
            edge_score = self.fcc(edge_emb)

            if t == 0:
                bce_loss = F.binary_cross_entropy(edge_score, y, reduction='none')
            else:
                bce_loss = torch.vstack([bce_loss, F.binary_cross_entropy(edge_score, y, reduction='none')])
            # bce_loss += self._cal_at_loss(pos_edge, y_pos)
            label_rectifier = self.dec(z_t, edge_index, sigmoid=True)
            label_rectifier = label_rectifier.unsqueeze(1)
            next_y_list.append((0.9*y+0.1*label_rectifier).detach())
            reg_loss += torch.norm(label_rectifier-edge_score, dim=1, p=2).mean()#0异常 1正常
            kld_loss += self._kld_gauss(enc_mean_t_sl, enc_std_t_sl, prior_mean_t_sl, prior_std_t_sl)
            recon_loss += self._recon_loss(z_t, x, edge_index) 
            
            all_z.append(z_t)
            all_node_idx.append(node_index)
            all_h.append(h_t_sl)
            score_list.append(edge_score)
        bce_loss = bce_loss.squeeze() 
        reg_loss /= dataloader.__len__()
        recon_loss /= dataloader.__len__()
        kld_loss /= dataloader.__len__()
        nce_loss = self.contrastive(all_z, all_node_idx)

        return bce_loss, reg_loss, recon_loss + kld_loss, nce_loss, next_y_list, h_t, score_list
    
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
        kld_element = torch.mean(torch.sum(1 + 2 * std_log - mean_in.pow(2) -
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

    def _recon_loss(self, z, x, pos_edge_index, neg_edge_index=None):        
        x_hat = self.linear(z)
        feature_loss = self.mse(x, x_hat)
        weight = torch.sigmoid(torch.exp(-torch.norm(z[pos_edge_index[0]] - z[pos_edge_index[1]], dim=1, p=2)))
        pos_loss = (-torch.log(self.dec(z, pos_edge_index) + self.EPS)*weight).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        weight = torch.sigmoid(torch.exp(torch.norm(z[neg_edge_index[0]] - z[neg_edge_index[1]], dim=1, p=2)))
        neg_loss = (-torch.log(1 - self.dec(z, neg_edge_index) + self.EPS)*weight).mean()
        return pos_loss + neg_loss + feature_loss


    def _compute_ev(self, data, normalization=None, is_undirected=False):
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'
        edge_weight = data.edge_attr
        if edge_weight is not None and edge_weight.numel() != data.num_edges:
            edge_weight = None

        edge_index, edge_weight = get_laplacian(data.edge_index, edge_weight, normalization, num_nodes=data.num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_weight, data.num_nodes)
        eig_fn = eigs
        if is_undirected and normalization != 'rw':
            eig_fn = eigsh
        lambda_max,ev = eig_fn(L, k=1, which='LM', return_eigenvectors=True)
        ev = torch.from_numpy(ev)
        return ev 
'''
    # def _l2_normalize(self, d):
    #     size = len(d.size())
    #     d = d.numpy()
    #     if size == 2:
    #         d /= (np.sqrt(np.sum(d ** 2, axis=(1))).reshape((-1, 1)) + 1e-16)
    #     if size == 3:
    #         d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    #     return torch.from_numpy(d)
    
    # def _cal_at_loss(self, edge_rep, label, weight = None, cal_loss = True):
    #     # total_rep = torch.vstack([pos_rep, neg_rep])
    #     edge_score = self.fcc(edge_rep).squeeze()
    #     label = label.float()
    #     if weight != None:
    #         weight = Variable(torch.pow(torch.abs(edge_score.sub(label)), self.gamma))
    #     at_loss = F.binary_cross_entropy(edge_score, label, weight)
    #     rep_grad = grad(at_loss, edge_rep, retain_graph=True)
    #     r_adv = torch.FloatTensor(self.at_eps * self._l2_normalize(rep_grad[0].data.cpu()))
    #     r_adv = Variable(r_adv.to(self.device))
    #     at_score = self.fcc(edge_rep + r_adv.detach()).squeeze()
    #     self.zero_grad()
    #     if cal_loss == False:
    #         return torch.abs(at_score - edge_score)
    #     if weight != None:
    #         weight = Variable(torch.pow(torch.abs(at_score.sub(label)), self.gamma))
    #     at_loss = self.at_alpha * F.binary_cross_entropy(at_score, label, weight)
    #     return at_loss
    
    # def _kl_with_logit(self, q, p, reduction="batchmean"):
    #     logq = torch.log(q)
    #     loss = F.kl_div(logq, p, reduction = reduction)
    #     return loss
    
    # def _cal_vat_loss(self, edge_index, node_emb):
    #     embed_size = node_emb.size()
    #     d = torch.Tensor(torch.Size(embed_size)).normal_().to(self.device)
    #     new_node_emb = node_emb + d
    #     edge_rep = new_node_emb[edge_index[0]] + new_node_emb[edge_index[1]]
    #     y_f = self.fcc(edge_rep).squeeze()
    #     for _ in range(self.vat_iter):
    #         d = self.vat_xi * self._l2_normalize(d.cpu())
    #         d = Variable(d.to(self.device), requires_grad=True)
    #         new_node_emb = node_emb + d
    #         edge_rep = new_node_emb[edge_index[0]] + new_node_emb[edge_index[1]]
    #         y_n = self.fcc(edge_rep).squeeze()
    #         delta_kl = self._kl_with_logit(y_n, y_f.detach())
    #         delta_kl.backward(retain_graph=True)
    #         d = d.grad.data.clone().cpu()
    #         self.zero_grad()
    #     d = self._l2_normalize(d.cpu())
    #     d = Variable(d.to(self.device))
    #     r_adv = self.vat_eps * d
    #     new_node_emb = node_emb + d
    #     edge_rep = new_node_emb[edge_index[0]] + new_node_emb[edge_index[1]]
    #     y_e = self.fcc(edge_rep).squeeze()
    #     vat_loss = self.vat_alpha * self._kl_with_logit(y_e, y_f.detach())
    #     return vat_loss
   
    # def _cal_vat_loss(self, neg_index, z, logits):
    #     node_emb = z
    #     noise = node_emb.data.new(node_emb.size()).normal_(0, 1)*1e-5
    #     noise.requires_grad_()
    #     new_node_emb = node_emb.data.detach() + noise
    #     neg_rep = new_node_emb[neg_index[0]] + new_node_emb[neg_index[1]]
    #     vat_score = self.fcc(neg_rep).squeeze()
    #     vat_loss = self._kl_with_logit(vat_score, logits.detach())        
    #     vat_grad = grad(vat_loss, noise, retain_graph=True)[0]
    #     #norm = vat_grad.norm()
        
    #     noise = noise + vat_grad * self.vat_xi
    #     noise = torch.FloatTensor(self.vat_eps * self._l2_normalize(noise.data.cpu()))
    #     noise = Variable(noise.to(self.device))
    #     node_emb = node_emb + noise.detach()
    #     neg_rep = node_emb[neg_index[0]] + node_emb[neg_index[1]]
    #     vat_score = self.fcc(neg_rep).squeeze()
    #     self.zero_grad()
    #     vat_loss = self.vat_alpha * self._kl_with_logit(vat_score, logits.detach())
    #     return vat_loss
'''  
        
        


class FCC(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(FCC,self).__init__()
        self.device = device
        self.linear = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=True, device=self.device),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.project(x)
    
