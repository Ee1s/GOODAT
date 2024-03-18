import warnings
warnings.filterwarnings("ignore")
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, global_mean_pool
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
import numpy as np
import math
from math import sqrt

class HCL(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, feat_dim, str_dim):
        super(HCL, self).__init__()

        self.embedding_dim = hidden_dim * num_gc_layers

        self.encoder_feat = Encoder_GIN(feat_dim, hidden_dim, num_gc_layers)
        self.encoder_str = Encoder_GIN(str_dim, hidden_dim, num_gc_layers)
        self.proj_head_feat_g = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_str_g = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                            nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_feat_n = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_str_n = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                            nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_b = nn.Sequential(nn.Linear(self.embedding_dim * 2, self.embedding_dim), nn.ReLU(inplace=True),
                                            nn.Linear(self.embedding_dim, self.embedding_dim))
        self.classifier = nn.Sequential(*([Linear(self.embedding_dim, 2)]))


        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def get_b(self, x_f, x_s, edge_index, batch, num_graphs):
        g_f, _ = self.encoder_feat(x_f, edge_index, batch)
        g_s, _ = self.encoder_str(x_s, edge_index, batch)
        b = self.proj_head_b(torch.cat((g_f, g_s), 1))
        return b

    def forward(self, x_f, edge_index, batch, num_graphs):

        g_f, n_f = self.encoder_feat(x_f, edge_index, batch)

        g_f = self.proj_head_feat_g(g_f)
        g_c = self.classifier(g_f)

        n_f = self.proj_head_feat_n(n_f)

        return g_f, n_f, g_c

    @staticmethod
    def scoring_b(b, cluster_result, temperature = 0.2):
        im2cluster, prototypes, density = cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density']

        batch_size, _ = b.size()
        b_abs = b.norm(dim=1)
        prototypes_abs = prototypes.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', b, prototypes) / torch.einsum('i,j->ij', b_abs, prototypes_abs)
        sim_matrix = torch.exp(sim_matrix / (temperature * density))

        v, id = torch.min(sim_matrix, 1)

        return v

    @staticmethod
    def calc_loss_b(b, index, cluster_result, temperature=0.2):
        im2cluster, prototypes, density = cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density']
        pos_proto_id = im2cluster[index].cpu().tolist()

        batch_size, _ = b.size()
        b_abs = b.norm(dim=1)
        prototypes_abs = prototypes.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', b, prototypes) / torch.einsum('i,j->ij', b_abs, prototypes_abs)
        sim_matrix = torch.exp(sim_matrix / (temperature * density))
        pos_sim = sim_matrix[range(batch_size), pos_proto_id]

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss + 1e-12)
        return loss

    @staticmethod
    def calc_loss_n(x, x_aug, batch, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        node_belonging_mask = batch.repeat(batch_size,1)
        node_belonging_mask = node_belonging_mask == node_belonging_mask.t()

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature) * node_belonging_mask
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-12)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-12)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        loss = global_mean_pool(loss, batch)

        return loss

    @staticmethod
    def calc_loss_g(x, x_aug, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        return loss

    def CELoss(self, y_pred, y_true):

        loss = F.cross_entropy(y_pred, y_true.long())
        return loss

    def cov(self, x, y):

        scaled_x = np.mean(x - np.mean(x), axis=1)
        scaled_y = np.mean(y - np.mean(y), axis=1)
        
        data = np.matrix([[scaled_x[i], scaled_y[i]] for i in range(len(scaled_x))])
        cov_matrix = np.dot(data, data.T) / (len(x) - 1)
        return cov_matrix

    def jointp(self, psc1, pnc1):
        psc1 =torch.normal(psc1)#.cpu().detach().numpy() 
        pnc1 = torch.normal(pnc1)#.cpu().detach().numpy() 
        psc = psc1.cpu().detach().numpy()
        pnc = pnc1.cpu().detach().numpy()
        conv = self.cov(psc,pnc)
        varsc1 = np.var(psc)
        varnc2 = np.var(pnc)
        rho = conv/(math.sqrt(varnc2)*math.sqrt(varsc1))
        expp = (-1/(2*(1-pow(rho,2)))) * (pow(psc,2)/pow(varsc1,2)-2*rho*(psc*pnc)/math.fabs(varsc1*varnc2)+pow(pnc,2)/pow(varnc2,2))
        pjoint2 = (1/2*math.pi)*(np.fabs(varsc1*varnc2)*np.sqrt(1-pow(rho,2)))*np.exp(expp)
        pjoint2 = torch.tensor(pjoint2, requires_grad=True)
        pjointmean = torch.mean(pjoint2, dim=1)

        return pjointmean #, pjoint2,

    def subgraphloss(self, detla_graph, label, args):
        loss_class_sc = self.CELoss(detla_graph, label)
        dkl = torch.mean(F.kl_div(torch.nn.LogSoftmax(dim=1)(detla_graph), torch.normal(detla_graph), reduction='none'),
                         dim=1)
        lsc = torch.mean(loss_class_sc + args.alpha * dkl)
        return lsc
    def csubgraphloss(self, T_detla_graph, label, args):
        loss_class_nc = self.CELoss(T_detla_graph, label)
        lnc = torch.mean(-loss_class_nc - args.beta * torch.mean(
            F.kl_div(torch.nn.LogSoftmax(dim=1)(T_detla_graph), torch.normal(T_detla_graph), reduction='none'), dim=1))
        return lnc



    def IBLoss(self, detla_graph, T_detla_graph, label, args):
        lsc = self.subgraphloss(detla_graph, label, args)
        lnc = self.csubgraphloss(T_detla_graph, label, args)
        ljoint = self.jointp(detla_graph,T_detla_graph)
        ljoint = torch.mean(ljoint)
        ibloss = lnc+lsc+ljoint
        return ibloss

class Encoder_GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder_GIN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            self.convs.append(conv)


    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)

        return x, torch.cat(xs, 1)


class Encoder_GCN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder_GCN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                conv = GCNConv(dim, dim)
            else:
                conv = GCNConv(num_features, dim)
            self.convs.append(conv)


    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        xpool = [global_mean_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)
