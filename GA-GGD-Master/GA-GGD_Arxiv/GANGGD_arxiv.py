import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import random
import copy
import torch_geometric
from ggd import GGD, Classifier
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import os
from sklearn import preprocessing as sk_prep
from gene import Generator

def aug_feature_dropout(input_feat, drop_percent=0.2):
    aug_input_feat = copy.deepcopy(input_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0
    return aug_input_feat


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def load_data_ogb(dataset, args):
    global n_node_feats, n_classes

    if args.data_root_dir == 'default':
        data = DglNodePropPredDataset(name=dataset)
    else:
        data = DglNodePropPredDataset(name=dataset, root=args.data_root_dir)

    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph):
    global n_node_feats

    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph


def main(args):
    cuda = True
    free_gpu_id = int(args.gpu)
    torch.cuda.set_device(args.gpu)
    # load and preprocess dataset
    if 'ogbn' not in args.dataset_name:
        data = load_data(args)
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        if hasattr(torch, 'BoolTensor'):
            train_mask = torch.BoolTensor(data.train_mask)
            val_mask = torch.BoolTensor(data.val_mask)
            test_mask = torch.BoolTensor(data.test_mask)
        else:
            train_mask = torch.ByteTensor(data.train_mask)
            val_mask = torch.ByteTensor(data.val_mask)
            test_mask = torch.ByteTensor(data.test_mask)
        in_feats = features.shape[1]
        n_classes = data.num_labels
        n_edges = data.graph.number_of_edges()
        g = data.graph
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        if args.self_loop:
            g.remove_edges_from(nx.selfloop_edges(g))
            g.add_edges_from(zip(g.nodes(), g.nodes()))
        g = DGLGraph(g)
    else:
        g, labels, train_mask, val_mask, test_mask, evaluator = load_data_ogb(args.dataset_name, args)
        g = preprocess(g)

        features = g.ndata['feat']
        labels = labels.T.squeeze(0)

        g, labels, train_idx, val_idx, test_idx, features = map(
            lambda x: x.to(free_gpu_id), (g, labels, train_mask, val_mask, test_mask, features)
        )
        in_feats = g.ndata['feat'].shape[1]
        n_classes = labels.T.max().item() + 1
        n_edges = g.num_edges()

    g = g.to(free_gpu_id)
    # create GGD model
    ggd = GGD(g,
              in_feats,
              args.n_hidden,
              args.n_layers,
              nn.PReLU(args.n_hidden),
              args.dropout,
              args.proj_layers,
              args.gnn_encoder,
              args.num_hop)
    generator = Generator(features,g,args.n_hidden,args.n_layers,nn.PReLU(args.n_hidden),args.dropout)

    if cuda:
        ggd.cuda()
        generator.cuda()

    ggd_optimizer = torch.optim.AdamW(ggd.parameters(),
                                      lr=args.ggd_lr,
                                      weight_decay=args.weight_decay)
    gene_optimizer = torch.optim.AdamW(generator.parameters(),
                                       lr=args.ggd_lr,
                                       weight_decay=args.weight_decay)
    b_xent = nn.BCEWithLogitsLoss()
    gene_b_xent = nn.BCEWithLogitsLoss()
    accs = []
    for it in range(args.iteration):
        cnt_wait = 0
        best = 1e9
        best_t = 0
        dur = []

        tag = str(int(np.random.random() * 10000000000))
        gene_x = generator.get_graph()
        for epoch in range(args.num_epochs):
            ggd.train()
            generator.eval()
            if epoch >= 3:
                t0 = time.time()

            ggd_optimizer.zero_grad()
            gene_optimizer.zero_grad()

            lbl_1 = torch.ones(1, g.num_nodes())
            lbl_2 = torch.zeros(1, g.num_nodes())
            lbl = torch.cat((lbl_1, lbl_2), 1).cuda()
            aug_feat = aug_feature_dropout(features, args.drop_feat)
            if it == 0:
                loss = ggd(aug_feat, lbl, None, b_xent)
            else:
                loss = ggd(aug_feat, lbl, gene_x, b_xent)
            loss.backward()
            ggd_optimizer.step()

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(ggd.state_dict(), 'pkl/best_ggd' + tag + '.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!')
                break

            if epoch >= 3:
                dur.append(time.time() - t0)

            print("Iteration {:05d} |Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | ".format(it,epoch, np.mean(dur), loss.item()))
        print('Disc_Training Completed.')
        print('Train Dis Loading {}th epoch'.format(best_t))

        ggd.load_state_dict(torch.load('pkl/best_ggd' + tag + '.pkl'))
        cnt_wait = 0
        gene_best = 1e9
        gene_best_t = 0
        gene_counts = 0
        gene_dur = []

        with torch.no_grad():
            feat = ggd.encode_for_gTrain(features)
        for epoch in range(args.num_epochs):
            ggd.train()
            generator.train()
            if epoch >= 3:
                t0 = time.time()
            ggd_optimizer.zero_grad()
            gene_optimizer.zero_grad()

            gene_x = generator.generate_graph()
            logits_g = ggd.encode(gene_x)
            lbl = torch.ones_like(logits_g)
            loss_gene = gene_b_xent(logits_g, lbl) - (torch.nn.functional.normalize(gene_x, dim=1) * feat).sum(1).mean()
            loss_gene.backward()
            gene_optimizer.step()

            if loss_gene < gene_best:
                gene_best = loss_gene
                gene_best_t = epoch
                cnt_wait = 0
                torch.save(generator.state_dict(), 'pkl/best_generator' + tag + '.pkl')
            else:
                cnt_wait += 1
            if cnt_wait == args.patience:
                print('Early stopping!')
                break
            if epoch >= 3:
                dur.append(time.time() - t0)
            print("Iteration {:05d} |Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | ".format(it,epoch, np.mean(dur), loss_gene.item()))
        print('Gene_Training Completed.')
        generator.load_state_dict(torch.load('pkl/best_generator' + tag + '.pkl'))

        # create classifier model
        classifier = Classifier(args.n_hidden, n_classes)
        # classifier = Classifier(features.size()[1], n_classes)
        if cuda:
            classifier.cuda()

        classifier_optimizer = torch.optim.AdamW(classifier.parameters(),
                                                 lr=args.classifier_lr,
                                                 weight_decay=args.weight_decay)

        # train classifier

        ggd.load_state_dict(torch.load('pkl/best_ggd' + tag + '.pkl'))

        # graph power embedding reinforcement
        l_embeds, g_embeds = ggd.embed(features, g)

        embeds = (l_embeds + g_embeds).squeeze(0)

        embeds = sk_prep.normalize(X=embeds.cpu().numpy(), norm="l2")

        embeds = torch.FloatTensor(embeds).cuda()
        # embeds = features
        dur = []
        best_acc, best_val_acc = 0, 0
        print('Testing Phase ==== Please Wait.')
        for epoch in range(args.n_classifier_epochs):
            classifier.train()
            if epoch >= 3:
                t0 = time.time()

            classifier_optimizer.zero_grad()
            preds = classifier(embeds)
            loss = F.nll_loss(preds[train_mask], labels[train_mask])
            loss.backward()
            classifier_optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            val_acc = evaluate(classifier, embeds, labels, val_mask)
            if epoch > 1000:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = evaluate(classifier, embeds, labels, test_mask)
                    if test_acc > best_acc:
                        best_acc = test_acc

        print("Valid Accuracy {:.4f}".format(best_val_acc))
        print("Test Accuracy {:.4f}".format(best_acc))

        with open(args.log_dir, 'a') as f:
            f.write('\n')
            f.write(' iteration: ' + str(it) + ' best_acc ' + str(best_acc) )
        accs.append(best_acc)
    return accs



if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='GANGGD_arxiv')
    register_data_args(parser)
    parser.add_argument("--iteration",type=int,default=10,help="Iteration")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--ggd-lr", type=float, default=0.0001,
                        help="ggd learning rate")
    parser.add_argument("--drop_feat", type=float, default=0.1,
                        help="feature dropout rate")
    parser.add_argument("--classifier-lr", type=float, default=0.05,
                        help="classifier learning rate")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=6000,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=1500,
                        help="number of hidden gcn units")
    parser.add_argument("--proj_layers", type=int, default=1,
                        help="number of project linear layers")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=500,
                        help="early stop patience condition")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--n_trails", type=int, default=5,
                        help="number of trails")
    parser.add_argument("--gnn_encoder", type=str, default='gcn',
                        help="choice of gnn encoder")
    parser.add_argument("--num_hop", type=int, default=10,
                        help="number of k for sgc")
    parser.add_argument('--data_root_dir', type=str,
                        default='/media/shaqserver/WS/GANGCL-IF/SLZ/GANGGD_arxiv/datasets',
                        help="data_root_dir")
    parser.add_argument('--dataset_name', type=str, default='ogbn-arxiv',
                        help='Dataset name: ogbn-arxiv')
    parser.add_argument('--seed', type=int, default=66666, help='seed')
    parser.add_argument('--log_dir', type=str, default='./log/logOgbn-arxiv.txt', help='seed')

    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)
    torch_geometric.seed.seed_everything(args.seed)
    with open(args.log_dir, 'a') as f:
        f.write('\n\n\n')
        f.write(str(args))
    for i in range(args.n_trails):
        main(args)

    # file_name = str(args.dataset_name)
    # f = open('result/' + 'result_' + file_name + '.txt', 'a')
    # f.write(str(args) + '\n')
    # f.write(mean_acc + '\n')
