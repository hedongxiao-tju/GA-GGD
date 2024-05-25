import torch
import torch.nn as nn
from gcn import GCN



class Generator(nn.Module):
    def __init__(self,features,g,hid_num,n_layers,activation,dropout):
        super(Generator,self).__init__()
        self.conv = GCN(g,1,hid_num,hid_num,n_layers,activation,dropout)
        self.features = features
        self.g = g
        self.generate_graph = self.generate_graph_gcn_generator

    def generate_graph_gcn_generator(self,l_data,blocks):
        return self.conv.forward2(l_data,blocks)

    def get_graph(self,l_data,blocks):
        self.eval()
        with torch.no_grad():
            x = self.generate_graph(l_data, blocks)
        return x.detach()

    def cuda(self):
        self.conv = self.conv.cuda()