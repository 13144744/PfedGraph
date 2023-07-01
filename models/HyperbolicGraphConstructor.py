from math import inf
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

from hgcn.layers import hyp_layers
from hgcn.layers.hyp_layers import HypLinear
from hgcn.manifolds import PoincareBall


class HyperbolicGraphConstructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, nclass, args):
        super(HyperbolicGraphConstructor, self).__init__()
        self.num_features = nnodes
        self.nhid = dim
        self.args = args
        self.c = args.c
        self.manifold = PoincareBall()
        self.use_bias = args.use_bias
        self.dropout = args.dropout
        self.act = torch.nn.ReLU()
        self.pooling_ratio = 0.5
        self.sample = True
        self.sparse = True
        self.sl = True
        self.lamb = 1.0
        self.alpha = 3
        self.device = device

        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)
        self.hyp_lin1 = HypLinear(self.manifold, dim, dim, self.c, 0.01, True)
        self.hyp_lin2 = HypLinear(self.manifold, dim, dim, self.c, 0.01, True)
        self.hyp_lin3 = HypLinear(self.manifold, 10, 10, self.c, 0.01, True)

    def forward(self, idx, dist_metrix, x):
        nodevec1 = self.emb1(idx)
        nodevec1 = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(nodevec1, self.c), self.c), self.c)
        nodevec2 = self.emb2(idx)
        nodevec2 = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(nodevec2, self.c), self.c), self.c)

        nodevec1 = self.hyp_lin1(nodevec1)
        nodevec2 = self.hyp_lin2(nodevec2)
        a1 = self.manifold.mobius_matvec(nodevec1, nodevec2, self.c)
        a2 = self.manifold.mobius_matvec(nodevec2, nodevec1, self.c)
        a = self.manifold.mobius_add(a1, -a2, c=self.c)

        adj = F.relu(torch.tanh(self.alpha * a))
        eye = (torch.eye(len(adj), len(adj)) * 0.0001).to(self.device)
        x = x.to(self.device)
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        adj = normalize_graph_adj(adj + eye, self.device)

        x = self.manifold.mobius_matvec(x.transpose(1, 0), adj, self.c)
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return x

    def loss(self, pred, label, param_metrix):
        loss = 0
        for i in range(len(pred)):
            mu = 1
            loss += mu * torch.norm(pred[i] - label[i]) ** 2
        return loss


def matrix2list(matrix):
    result = []
    N = len(matrix)
    for i in range(N):
        for j in range(N):
            if matrix[i][j] and matrix[i][j] != inf:
                result.append((i, j))
    result = torch.tensor(result).transpose(-1, -2)
    return result


def normalize_graph_adj(mx, device):
    """Row-normalize sparse matrix"""
    mx = mx.cpu()
    rowsum = np.array(mx.detach().sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx.detach())
    mx = torch.tensor(mx)
    return mx.to(device)
