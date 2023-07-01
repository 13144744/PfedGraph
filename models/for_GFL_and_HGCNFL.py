import numpy as np
import scipy.sparse as sp
import torch

from models.HyperbolicGraphConstructor import HyperbolicGraphConstructor


def generate_param(param_metrix, args, gc_model, subgraph_size, train_size, server_weight):
    dist_metrix = torch.zeros((len(param_metrix), len(param_metrix)))
    avg_metrix = torch.sum(torch.stack([torch.mul(param_metrix[i], train_size[i]) for i in range(len(param_metrix))]),
                           dim=0)
    total_metrix = []
    for i in range(len(param_metrix)):
        for j in range(len(param_metrix)):
            dist_metrix[i][j] = torch.nn.functional.pairwise_distance(
                param_metrix[i].view(1, -1), param_metrix[j].view(1, -1), p=2).clone().detach()
        total_metrix.append(avg_metrix)
    dist_metrix = torch.nn.functional.normalize(dist_metrix).to(args.device)
    idx = torch.arange(args.num_clients).to(args.device)
    gc_model = "HGCN"
    if gc_model == "HGCN":
        hidden = 64
        gc = HyperbolicGraphConstructor(args.num_clients, subgraph_size, hidden, args.device, 10, args).to(args.device)
        optimizer = torch.optim.SGD(gc.parameters(), lr=args.gc_lr, weight_decay=args.wd)
    # gc_epoch = 10
    for e in range(args.gc_epoch):
        optimizer.zero_grad()
        weights = gc(idx, dist_metrix, param_metrix)
        optimizer.step()

    weights = gc(idx, dist_metrix, param_metrix)
    weights = weights.to("cpu")

    # adj = adj + ones
    return weights, avg_metrix


def normalize_graph_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
