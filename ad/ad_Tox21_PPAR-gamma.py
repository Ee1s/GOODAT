def warn(*args, **kwargs):
    pass
import warnings
warnings.filterwarnings("ignore")
import math
from math import sqrt
warnings.warn = warn
from model import HCL
from data_loader import *
import argparse
import numpy as np
import torch
import random
import faiss
import sklearn.metrics as skm
import torch_geometric
from torch.nn.parameter import Parameter
import scipy.stats
from sklearn.cluster import KMeans, SpectralClustering
import torch.nn.functional as F
from torch_sparse import coalesce, SparseTensor
# from modify_adj import EdgeAgentz/
from sklearn.metrics import roc_auc_score as sk_roc_auc, mean_squared_error, \
    accuracy_score, average_precision_score, mean_absolute_error, f1_score
import torch_sparse
from torch_sparse import coalesce

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_type', type=str, default='ad', choices=['oodd', 'ad'])
    parser.add_argument('-lr_feat',  type=float, default=0.02)
    parser.add_argument('-lr_adj',  type=float, default=0.02)
    parser.add_argument('-alpha', type=float, default=0.8)
    parser.add_argument('-beta', type=float, default=0.006)
    parser.add_argument('-num_epoch', type=int, default=200)
    parser.add_argument('-lr', type=float, default=0.00005)
    parser.add_argument('-num_pertrain',  type=int, default=100)
    parser.add_argument('-DS', help='Dataset', default='Tox21_PPAR-gamma')
    parser.add_argument('-DS_ood', help='Dataset', default='Tox21_PPAR-gamma')
    parser.add_argument('-DS_pair', default=None)
    parser.add_argument('-rw_dim', type=int, default=16)
    parser.add_argument('-dg_dim', type=int, default=16)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-batch_size_test', type=int, default=9999)
    parser.add_argument('-num_layer', type=int, default=5)
    parser.add_argument('-hidden_dim', type=int, default=16)
    parser.add_argument('-num_trial', type=int, default=5)
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-is_adaptive', type=int, default=1)
    parser.add_argument('-num_cluster', type=int, default=2)

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)



def reluDerivative(x):
    x[x <= 0] = 1e-10
    # x[x > 0] = x
    return x

def flip_array(array):
    flipped_array = []
    for element in array:
        flipped_array.append(1 - element)  
    return flipped_array



def learn_graph(data, model, args):
    data = data.to(device)
    model.eval()
    x = data.x
    edge_index = data.edge_index
    batch = data.batch
    g_f, n_f, g_c = model(x, edge_index, batch, data.num_graphs)
    g_cs = g_c.softmax(dim=1).cpu()
    g_cs = np.argmax(g_cs, axis=1)

    #------tuning node features----------#
    nnodes = x.shape[0]
    d = x.shape[1]


    #-----------tuning featurers
    delta_feat = Parameter(torch.FloatTensor(nnodes, d).to(device))
    delta_feat.data.fill_(1e-7)
    delta_feat.requires_grad = True
    optimizer_feat = torch.optim.Adam([delta_feat], lr=args.lr_feat)

   
    #------==-tuning adjacent---------#
    edge_weight = torch.ones(edge_index.shape[1])  
    perturbed_edge_weight = torch.full_like(edge_weight, 1e-7, dtype=torch.float32, requires_grad=True)
    edge_weight = edge_weight + perturbed_edge_weight
    edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]  # 2-1
    perturbed_edge_weight.requires_grad = True
    optimizer_adj = torch.optim.Adam([perturbed_edge_weight], lr=args.lr_adj)
    perturbed_edge_weight = perturbed_edge_weight.to(device)
    edge_weight = edge_weight.to(device)

    #---------training----------------#
    for loopfeat in range(args.num_epoch):
        #----------------only--feature----------------#
        optimizer_feat.zero_grad()
        optimizer_adj.zero_grad()
        if edge_weight is not None:
            t_adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                               sparse_sizes=2 * x.shape[:1]).t()
            c_adj = SparseTensor.from_edge_index(edge_index, perturbed_edge_weight,
                                               sparse_sizes=2 * x.shape[:1]).t() # in case it is directed...
        tg_f, tn_f, tg_c = model(x + delta_feat, t_adj, batch, data.num_graphs)
        cg_f, cn_f, cg_c = model(delta_feat, c_adj, batch, data.num_graphs)
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]
        loss = model.IBLoss(tg_f, cg_f, g_cs.to(device), args)
       # loss.requires_grad_(True)
        loss.backward()
        optimizer_feat.step()
        optimizer_adj.step()

    yscore = torch.nn.functional.cross_entropy(tg_f, g_cs.to(device), reduction='none')+args.alpha*torch.mean(F.kl_div(torch.nn.LogSoftmax(dim=1)(tg_f), torch.normal(tg_f), reduction='none'),
                         dim=1)
    return yscore



if __name__ == '__main__':
    setup_seed(0)

    args = arg_parse()

    if args.exp_type == 'ad':
        if args.DS.startswith('Tox21'):
            dataloader, dataloader_test, meta = get_ad_dataset_Tox21(args)
        else:
            splits = get_ad_split_TU(args, fold=args.num_trial)

    if args.exp_type == 'oodd':
        dataloader, dataloader_test, meta, dataloader_train = get_ood_dataset(args) #收集数据
    elif args.exp_type == 'ad' and not args.DS.startswith('Tox21'):
        dataloader, dataloader_test, meta = get_ad_dataset_TU(args, splits[0])

    dataset_num_features = meta['num_feat']
    n_train = meta['num_train']

    # if trial == 0:
    print(args)
    print('================')
    print('Exp_type: {}'.format(args.exp_type))
    print('DS: {}'.format(args.DS_pair if args.DS_pair is not None else args.DS))
    print('num_features: {}'.format(dataset_num_features))
    print('num_structural_encodings: {}'.format(args.dg_dim + args.rw_dim))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_layer))
    print('================')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HCL(args.hidden_dim, args.num_layer, dataset_num_features, args.dg_dim+args.rw_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model = torch.load('ad_Tox21_PPAR-gamma.pt')

    aucs = []
    print('--------Training TTA model----- ')
    for trial in range(args.num_trial):    #TTA
        model.eval()
        setup_seed(trial + 1)
        if args.exp_type == 'oodd':
            dataloader, dataloader_test, meta = get_ood_dataset(args)
        elif args.exp_type == 'ad' and not args.DS.startswith('Tox21'):
            dataloader, dataloader_test, meta = get_ad_dataset_TU(args, splits[trial])
        score = []
        y_score_all = []
        y_true_all = []
        #print('[TTA] Config TTA model, ', trial)

        # train the model
        for data in dataloader_test:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

            y_score = learn_graph(data, model, args) 
            y_true = data.y

            y_score_all = y_score_all + y_score.detach().cpu().tolist()
            y_true_all = y_true_all + y_true.detach().cpu().tolist()

        auc = skm.roc_auc_score(y_true_all, y_score_all)
        print('[TTA] Epoch: {:03d} | AUC:{:.4f}'.format(trial, auc))
        aucs.append(auc)

    avg_auc = np.mean(aucs)*100
    std_auc = np.std(aucs)*100

    print('[FINAL RESULT] AVG_AUC: {:.2f}+-{:.2f}'.format(avg_auc, std_auc))
