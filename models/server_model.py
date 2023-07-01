import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool, GCNConv, GINConv, GATConv, SAGEConv

from hgcn.layers.hyp_layers import GCN
from hgcn.layers.hyplayers import HgpslPool
from hgcn.layers.layers import Linear
from hgcn.layers import hyp_layers, hyplayers
from hgcn.manifolds.poincare import PoincareBall

from layers import HGPSLPool


class server_GCN(nn.Module):
    def __init__(self, args, num_features, nhid, num_classes):
        super(server_GCN, self).__init__()
        self.args = args
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.dropout_ratio = args.dropout

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, self.num_classes)


class serverGAT(torch.nn.Module):
    def __init__(self, n_feat, n_hid, n_class, nlayer, dropout, is_concat=True, leaky_relu_negative_slope=0.2):
        super(serverGAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(n_feat, n_hid)
        self.conv2 = GATConv(n_hid, n_hid)
        self.conv3 = GATConv(n_hid, n_hid)
        self.post1 = torch.nn.Sequential(torch.nn.Linear(n_hid, n_hid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(n_hid, n_class))


class serverGraphSAGE(torch.nn.Module):
    def __init__(self, n_feat, n_hid, n_class, nlayer, dropout, is_concat=True, leaky_relu_negative_slope=0.2):
        super(serverGraphSAGE, self).__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(n_feat, n_hid)
        self.conv2 = SAGEConv(n_hid, n_hid)
        self.conv3 = SAGEConv(n_hid, n_hid)
        self.post1 = torch.nn.Sequential(torch.nn.Linear(n_hid, n_hid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(n_hid, n_class))
