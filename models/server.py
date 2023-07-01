import random
from copy import deepcopy
from math import tan
from dtaidistance import dtw
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering
from torch import nn
from hgcn.utils.math_utils import artanh, tanh
from hgcn.manifolds import PoincareBall
from models.for_GFL_and_HGCNFL import generate_param
from utils.data_utils import *
from utils.util import *


class Server_Net(nn.Module):
    def __init__(self, model, device, args):
        super().__init__()
        self.model = model.to(device)
        self.c = args.c
        self.manifold = PoincareBall()
        self.W = {key: real for key, real in self.model.named_parameters()}
        self.dW = {key: real for key, real in self.model.named_parameters()}
        self.model_cache = []
        self.args = args

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            self.W[k].data = torch.div(
                torch.sum(
                    torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]),
                    dim=0), total_size).clone()

    def hgcn_aggregate_weights(self, args, gc_model, selected_clients):
        total_size = 0
        Ws, dWs, grads = [], [], []
        train_size = []
        for client in selected_clients:
            total_size += client.train_size
        for client in selected_clients:
            W, dW, grad = {}, {}, {}
            for k in self.W.keys():
                W[k] = client.W[k]
            Ws.append(W)
            train_size.append(client.train_size / total_size)
        for k in self.W.keys():
            self.W[k].data = torch.div(
                torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]),
                          dim=0), total_size).clone()
        server_weight = self.W
        client_weight, avg_metrix = graph_dic(Ws, args, gc_model, total_size, train_size, server_weight)
        return client_weight, avg_metrix


def eval_server(model, test_loader, device):
    model.eval()
    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for databatch in test_loader:
        databatch.to(device)
        # print("eval_local")
        adj = reset_batch_adj(databatch)
        pred = model(databatch, adj)
        label = databatch.y
        loss = model.loss(pred, label)
        total_loss += loss.item() * databatch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += databatch.num_graphs
    return total_loss / ngraphs, acc_sum / ngraphs


def graph_dic(models_dic, args, gc_model, total_size, train_size, server_weight):
    keys = []
    key_shapes = []
    param_metrix, total_metrix = [], []
    for model in models_dic:
        param_metrix.append(sd_matrixing(model).clone().detach())
    param_metrix = torch.stack(param_metrix)
    avg_metrix = torch.sum(torch.stack([torch.mul(param_metrix[i], train_size[i]) for i in range(len(param_metrix))]),
                           dim=0)
    for i in range(len(param_metrix)):
        total_metrix.append(avg_metrix)
    total_metrix = torch.stack(total_metrix, 0)
    for key, param in models_dic[0].items():
        keys.append(key)
        key_shapes.append(list(param.data.shape))

    # constract adj
    subgraph_size = min(30, args.num_clients)
    aggregated_param, avg_metrix = generate_param(param_metrix, args, gc_model, subgraph_size, train_size, server_weight)
    aggregated_param = torch.tensor(aggregated_param.cpu().detach().numpy())
    new_param_matrix = (args.gc_ratio * aggregated_param) + ((1 - args.gc_ratio) * param_metrix)
    for i in range(len(models_dic)):
        pointer = 0
        for k in range(len(keys)):
            num_p = 1
            for n in key_shapes[k]:
                num_p *= n
            models_dic[i][keys[k]] = new_param_matrix[i][pointer:pointer + num_p].reshape(key_shapes[k])
            pointer += num_p
    return models_dic, avg_metrix
