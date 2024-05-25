import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SGConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 bias = True,
                 weight=True):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hidden, weight = weight, bias = bias, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, weight=weight, bias=bias, activation=activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
            # h = self.bns[i](h)
        return h

