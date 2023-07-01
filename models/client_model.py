import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool, GCNConv, GINConv, SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from hgcn.layers.hyp_layers import GCN
from hgcn.layers.hyplayers import HgpslPool
from hgcn.layers.layers import Linear
from hgcn.layers import hyp_layers, hyplayers
from layers import HGPSLPool
from hgcn.manifolds.poincare import PoincareBall


def edge_to_adj(edge_index, x):
    row, col = edge_index
    xrow, xcol = x[row], x[col]
    cat = torch.cat([xrow, xcol], dim=1).sum(dim=-1).div(2)
    weights = (torch.cat([x[row], x[col]], dim=1)).sum(dim=-1).div(2)
    adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
    adj[row, col] = weights
    return adj


class client_GCN(nn.Module):
    def __init__(self, args, num_features, nhid, num_classes):
        super(client_GCN, self).__init__()
        self.args = args
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.dropout_ratio = args.dropout

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(global_add_pool(x, batch))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GAT(torch.nn.Module):
    def __init__(self, n_feat, n_hid, n_class, nlayer, dropout, is_concat=True, leaky_relu_negative_slope=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(n_feat, n_hid)
        self.conv2 = GATConv(n_hid, n_hid)
        self.conv3 = GATConv(n_hid, n_hid)
        self.post1 = torch.nn.Sequential(torch.nn.Linear(n_hid, n_hid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(n_hid, n_class))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)

        x = global_add_pool(x, batch)
        x = self.post1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GraphSAGE(torch.nn.Module):
    def __init__(self, n_feat, n_hid, n_class, nlayer, dropout, is_concat=True, leaky_relu_negative_slope=0.2):
        super(GraphSAGE, self).__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(n_feat, n_hid)
        self.conv2 = SAGEConv(n_hid, n_hid)
        self.conv3 = SAGEConv(n_hid, n_hid)
        self.post1 = torch.nn.Sequential(torch.nn.Linear(n_hid, n_hid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(n_hid, n_class))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)

        x = global_add_pool(x, batch)
        x = self.post1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
