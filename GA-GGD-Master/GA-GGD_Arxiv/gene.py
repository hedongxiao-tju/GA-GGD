import torch
import torch.nn as nn
from gcn import GCN



class Generator(nn.Module):
    def __init__(self,features,g,hid_num,n_layers,activation,dropout):
        super(Generator,self).__init__()
        self.x = torch.rand(g.num_nodes()).view(-1,1)
        self.conv = GCN(g,1,hid_num,hid_num,n_layers,activation,dropout)
        self.features = features
        self.g = g
        self.generate_graph = self.generate_graph_gcn_generator

    def generate_graph_gcn_generator(self):
        return self.conv(self.x)

    def get_graph(self):
        self.eval()
        with torch.no_grad():
            x = self.generate_graph()
        return x.detach()

    def cuda(self):
        self.x = self.x.cuda()
        self.conv = self.conv.cuda()