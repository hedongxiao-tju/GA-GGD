#!/usr/bin/env python
# coding: utf-8
from sklearn.manifold import TSNE
from GCL.eval import get_split, LREvaluator

def test_on_class_multi(z, label, train_ratio=0.1, test_ratio=0.8, test_num=10):
    r = torch.zeros(test_num)
    for num in range(test_num):
        split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
        result = LREvaluator(num_epochs=10000)(z, label, split)
        r[num] = result['micro_f1']
    print('mean:', str(r.mean()), 'std:', str(r.std()))
    return r.mean(), r.std()
        
from tqdm import tqdm
import torch
def train_disc(nb_epochs, nb_nodes, features, sp_adj, drop_prob, 
               ggd, generator,patience,
               optimiser_disc, optimiser_gene,
               b_xent, tag, first_init):
    
    gene_x = generator.get_graph()
    best = 1e9
    best_t = 0
    avg_time = 0
    counts = 0
    cnt_wait = 0
    with tqdm(total=nb_epochs, desc='(T)') as pbar:
        for epoch in range(nb_epochs):
            ggd.train()
            generator.eval()
            optimiser_disc.zero_grad()
            optimiser_gene.zero_grad()

            aug_fts = aug_feature_dropout(features, drop_prob) # augmentation on features
            idx = np.random.permutation(nb_nodes)
            shuf_fts = aug_fts[idx, :]  # shuffled embeddings / corruption
                
            lbl = torch.ones(nb_nodes*2)
            lbl[nb_nodes:] = 0

            if torch.cuda.is_available():
                shuf_fts = shuf_fts.cuda()
                aug_fts = aug_fts.cuda()
                lbl = lbl.cuda()

            logits_1 = ggd(aug_fts, shuf_fts, None if first_init else gene_x, sp_adj)
            loss_disc = b_xent(logits_1, lbl)
            if loss_disc < best:
                best = loss_disc
                best_t = epoch
                cnt_wait = 0
                torch.save(ggd.state_dict(), 'pkl/best_dgi' + tag + '.pkl')
            else:
                cnt_wait += 1
            if cnt_wait == patience:
                print('Early stopping!')
                break
            loss_disc.backward()
            optimiser_disc.step()
            pbar.set_postfix({'loss': loss_disc.item()})
            pbar.update()
            
def train_gene(nb_epochs, nb_nodes, features, sp_adj, drop_prob, 
               ggd, generator,patience,
               optimiser_disc, optimiser_gene,
               b_xent, tag):
    best = 1e9
    best_t = 0
    avg_time = 0
    counts = 0
    cnt_wait = 0
    with torch.no_grad():
        feat = ggd.encode_for_gTrain(features, sp_adj)
        
    with tqdm(total=nb_epochs, desc='(T)') as pbar:
        for epoch in range(nb_epochs):
            ggd.train()
            generator.train()
            optimiser_disc.zero_grad()
            optimiser_gene.zero_grad()
            
            gene_x = generator.generate_graph()
            logits_g = ggd.encode(gene_x, sp_adj)
            lbl = torch.ones_like(logits_g)
            loss_gene = b_xent(logits_g, lbl) - (torch.nn.functional.normalize(gene_x, dim=1)*feat).sum(1).mean()
            
            if loss_gene < best:
                best = loss_gene
                best_t = epoch
                cnt_wait = 0
                torch.save(generator.state_dict(), 'pkl/best_dgi_generator' + tag + '.pkl')
            else:
                cnt_wait += 1
            if cnt_wait == patience:
                print('Early stopping!')
                break
            loss_gene.backward()
            optimiser_gene.step()
            pbar.set_postfix({'loss': loss_gene.item()})
            pbar.update()
            
import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.nn import GCNConv

class Generator(nn.Module):
    def __init__(self, generate_method, data_x, data_adj, hid_num, self_loop_for_adj):
        assert generate_method in ['gcn_generator']
        self.generate_method = generate_method
        super(Generator, self).__init__()
        if generate_method == 'gcn_generator':
            self.x = torch.rand(data_x.size()[0]).view(-1,1)
            self.gene1 = GCNConv(1, hid_num)
            self.gene2 = GCNConv(hid_num, hid_num)
            self.act = nn.PReLU()
            self.adj = data_adj
            self.generate_graph = self.generate_graph_gcn_generator
            
    def generate_graph_gcn_generator(self):
        return self.gene2(self.act(self.gene1(self.x, self.adj)), self.adj)
    
    def get_graph(self):
        self.eval()
        with torch.no_grad():
            x = self.generate_graph()
            x = x.detach()
        return x
    
    def cuda(self):
        self.x = self.x.cuda()
        self.gene1 = self.gene1.cuda()
        self.gene2 = self.gene2.cuda()
        self.act = self.act.cuda()
        self.adj = self.adj.cuda()
        
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import time
from models import LogReg
from utils import process
import os
import copy
import random
import argparse
import sys
from Dataset_Load import load_dataset
from torch_geometric.nn import GCNConv
import warnings
from torch_geometric.typing import torch_sparse
import torch_geometric
 
class GGD(nn.Module):
    def __init__(self, n_in, n_h, activator):
        super(GGD, self).__init__()
        self.gcn = GCNConv(n_in, n_h)
        self.act = activator()
        self.lin = nn.Linear(n_h, n_h)

    def forward(self, seq1, seq2, h_3, edge_index):
        h_1 = self.act(self.gcn(seq1, edge_index))
        h_2 = self.act(self.gcn(seq2, edge_index))
        h_2_copy = h_2.clone()
        h_1_copy = h_1.clone()
        if h_3 != None:
            s = torch.rand(h_1.size()[0])
            h_2[s>0.5] = h_3[s>0.5]
        sc_1 = ((self.lin(h_1)).sum(1))
        sc_2 = ((self.lin(h_2)).sum(1))
        
        logits = torch.cat((sc_1, sc_2))
        return logits
    
    def encode(self, seq, edge_index):        
        sc = ((self.lin(seq)).sum(1))
        return sc

    def embed(self, seq, edge_index, adj, Globalhop):
        self.eval()
        with torch.no_grad():
            h_1 = self.act(self.gcn(seq, edge_index))
            h_2 = h_1.clone()
            for i in range(Globalhop):
                h_2 = adj @ h_2 
            return h_1.detach(), h_2.detach()
        
    def encode_for_gTrain(self, seq, edge_index):
        self.eval()
        with torch.no_grad():
            h_1 = self.act(self.gcn(seq, edge_index))
            return h_1

def aug_feature_dropout(input_feat, drop_percent=0.2):
    aug_input_feat = copy.deepcopy(input_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0
    return aug_input_feat

def adj_norm(adj_t):
    deg = torch.sparse.sum(adj_t, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    adj_t = adj_t * deg_inv_sqrt.view(1, -1)
    adj_t = adj_t * deg_inv_sqrt.view(-1, 1)
    return adj_t   

def run(args):
    with open(args.log_dir, 'a') as f:
        f.write('\n\n\n')
        f.write(str(args))
    n_trails = args.n_trials
    free_gpu_id = args.GPU_ID
    torch.cuda.set_device(int(free_gpu_id))
    dataset = args.dataset
    data_dir = args.datadir
    nb_epochs = args.nb_epochs
    patience = args.patience
    lr = args.lr
    wd = args.wd
    drop_prob = args.drop_prob
    hid_units = args.hid_units
    num_hop = args.num_hop
    activator = nn.PReLU
    generate_method = args.generate_method
    iteration = args.iteration
    torch_geometric.seed.seed_everything(args.seed)
    seed = args.seed
    
    for trail in range(n_trails):
        data = load_dataset(dataset, data_dir)[0]
        features = data.x
        nb_nodes = data.x.size()[0]
        sp_adj = data.edge_index
        labels = data.y
        nb_classes = labels.size()[0]
        ft_size = features.size()[1]
        self_loop_for_adj = torch.Tensor([i for i in range(nb_nodes)]).unsqueeze(0)
        self_loop_for_adj = torch.concat([self_loop_for_adj, self_loop_for_adj], dim=0)
        slsp_adj = torch.concat([data.edge_index, self_loop_for_adj], dim=1)
        slsp_adj = torch.sparse.FloatTensor(slsp_adj.long(), torch.ones(slsp_adj.size()[1]),
                                            torch.Size([nb_nodes, nb_nodes]))
        slsp_adj = adj_norm(slsp_adj)
        generator = Generator(generate_method, features, sp_adj, hid_units, self_loop_for_adj)
        ggd = GGD(ft_size, hid_units, activator)
        optimiser_disc = torch.optim.Adam(ggd.parameters(), lr=lr, weight_decay=wd)
        optimiser_gene = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=wd)
        
        if torch.cuda.is_available():
            ggd.cuda()
            features = features.cuda()
            sp_adj = sp_adj.cuda()
            slsp_adj = slsp_adj.cuda()
            labels = labels.cuda()
            generator.cuda()  
        b_xent = nn.BCEWithLogitsLoss()
        tag = dataset+'_'+str(time.time())+'_'+str(trail)
        
        for it in range(iteration):

            train_disc(nb_epochs, nb_nodes, features, sp_adj, drop_prob, 
                       ggd, generator,patience,
                       optimiser_disc, optimiser_gene,
                       b_xent, tag, True if it==0 else False)
            ggd.load_state_dict(torch.load('pkl/best_dgi' + tag + '.pkl'))

            train_gene(int(nb_epochs/10), nb_nodes, features, sp_adj, drop_prob, 
                       ggd, generator,patience,
                       optimiser_disc, optimiser_gene,
                       b_xent, tag)
            generator.load_state_dict(torch.load('pkl/best_dgi_generator' + tag + '.pkl'))

            ggd.eval()
            or_embeds, pr_embeds = ggd.embed(features, sp_adj, slsp_adj, num_hop)
            embeds = or_embeds + pr_embeds
            m, r = test_on_class_multi(embeds, labels, train_ratio=0.1, test_ratio=0.8)
            with open(args.log_dir, 'a') as f:
                f.write('\n')
                f.write('NTRAIL: '+ str(trail) +' iteration: '+str(it)+' mean: '+str(m)+' std: '+ str(r))

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    #setting arguments
    parser = argparse.ArgumentParser('GA-GGD')
    parser.add_argument('--nb_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=500, help='Patience')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--drop_prob', type=float, default=0.2, help='drop percent')
    parser.add_argument('--hid_units', type=int, default=512, help='representation size')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name: Cora, Citeseer, PubMed, CS, Photo, Computers')
    parser.add_argument('--datadir', type=str, default='./datasets/', help='./data/dir/')
    parser.add_argument('--num_hop', type=int, default=10, help='graph view hop num')
    parser.add_argument('--n_trials', type=int, default=4, help='number of trails')
    parser.add_argument('--GPU_ID', type=int, default=0, help='The GPU ID')
    parser.add_argument('--generate_method', type=str, default='gcn_generator', help='gcn_generator')
    parser.add_argument('--iteration', type=int, default=7, help='iteration')
    parser.add_argument('--seed', type=int, default=66666, help='seed')
    parser.add_argument('--log_dir', type=str, default='./log/logCora.txt', help='seed')
    
    args = parser.parse_args()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    run(args)




