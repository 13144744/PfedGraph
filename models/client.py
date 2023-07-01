import time

from hgcn.manifolds import PoincareBall
from utils.data_utils import *
import torch.nn.functional as F
from torch import nn
from utils.util import *


def RMSE_error(pred, gold):
    return np.sqrt(np.mean((pred - gold) ** 2))


class Client_Net(nn.Module):
    def __init__(self, model, client_id, client_name, train_size, dataLoader, optimizer, args):
        super().__init__()
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args
        self.manifold = PoincareBall()
        self.c = args.c

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

        self.gconvNames = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.

    def download_from_server(self, server):
        self.gconvNames = server.W.keys()
        for k in server.W:
            if "hyperbolic_bias" in k:
                self.W[k].data = self.manifold.proj(
                    self.manifold.expmap0(self.manifold.proj_tan0(server.W[k].data.clone(), c=self.c), c=self.c),
                    c=self.c).clone()
            else:
                self.W[k].data = server.W[k].data.clone()

    def download_personalize_model_from_server(self, W):
        for k in W:
            self.W[k].data = W[k].data.clone()

    def reset(self):
        copy(target=self.W, source=self.W_old, keys=self.gconvNames)

    def local_HGCNFL_train(self, local_epoch, avg_metrix):
        """ For PFedGraph """
        copy(target=self.W_old, source=self.W, keys=self.gconvNames)

        train_stats = train_HGCNFL(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device, self.id,
                                   self.W_old, self.W, avg_metrix)

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W.items() and self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

        return train_stats["trainingLosses"][0], train_stats["trainingAccs"][0]

    def evaluate(self):
        return eval_local(self.model, self.dataLoader['test'], self.args.device)


def copy(target, source, keys):
    for name in keys:
        target[name].data = source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])


def calc_gradsNorm(gconvNames, Ws):
    grads_conv = {k: Ws[k].grad for k in gconvNames}
    convGradsNorm = torch.norm(flatten(grads_conv)).item()
    return convGradsNorm


def train_client(model, dataloaders, optimizer, local_epoch, device, clientNo):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0
        acc_sum = 0
        for _, batch in enumerate(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            ngraphs += batch.num_graphs
        total_loss /= ngraphs
        acc = acc_sum / ngraphs

        loss_v, acc_v = eval_local(model, val_loader, device)
        loss_tt, acc_tt = eval_local(model, test_loader, device)

        losses_train.append(total_loss)
        accs_train.append(acc)
        losses_val.append(loss_v)
        accs_val.append(acc_v)
        losses_test.append(loss_tt)
        accs_test.append(acc_tt)

    return {'trainingLosses': losses_train, 'trainingAccs': accs_train,
            'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test}


def train_HGCNFL(model, dataloaders, optimizer, local_epoch, device, clientNo, W_old, W_new,
                 avg_metrix):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0
        acc_sum = 0
        for _, batch in enumerate(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            m1 = list_matrixing(W_old).reshape(1, -1)
            if avg_metrix == None:
                loss = model.loss(pred, label)
            else:
                m2 = avg_metrix.to(device)
                loss = model.loss(pred, label) + 0.01 * torch.norm(m1 - m2) ** 2
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            ngraphs += batch.num_graphs
        total_loss /= ngraphs
        acc = acc_sum / ngraphs

        loss_v, acc_v = eval_HGCNFL(model, val_loader, device, avg_metrix, W_old)
        loss_tt, acc_tt = eval_HGCNFL(model, test_loader, device, avg_metrix, W_old)

        losses_train.append(total_loss)
        accs_train.append(acc)
        losses_val.append(loss_v)
        accs_val.append(acc_v)
        losses_test.append(loss_tt)
        accs_test.append(acc_tt)

    return {'trainingLosses': losses_train, 'trainingAccs': accs_train,
            'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test}


def eval_HGCNFL(model, test_loader, device, avg_metrix, W_old):
    model.eval()
    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            label = batch.y
            m1 = list_matrixing(W_old).reshape(1, -1)
            if avg_metrix == None:
                loss = model.loss(pred, label)
            else:
                m2 = avg_metrix.to(device)
                loss = model.loss(pred, label) + 0.01 * torch.norm(m1 - m2) ** 2
        total_loss += loss.item() * batch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += batch.num_graphs
    return total_loss / ngraphs, acc_sum / ngraphs


def eval_local(model, test_loader, device):
    model.eval()
    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
        total_loss += loss.item() * batch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += batch.num_graphs
    return total_loss / ngraphs, acc_sum / ngraphs


def _prox_term(model, gconvNames, Wt):
    prox = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        # only add the prox term for sharing layers (gConv)
        if name in gconvNames:
            prox = prox + torch.norm(param - Wt[name]).pow(2)
    return prox


def train_gc_prox(model, dataloaders, optimizer, local_epoch, device, gconvNames, Ws, mu, Wt):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    convGradsNorm = []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0
        acc_sum = 0
        for _, batch in enumerate(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label) + mu / 2. * _prox_term(model, gconvNames, Wt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            ngraphs += batch.num_graphs
        total_loss /= ngraphs
        acc = acc_sum / ngraphs

        loss_v, acc_v = eval_local(model, val_loader, device)
        loss_tt, acc_tt = eval_local(model, test_loader, device)

        losses_train.append(total_loss)
        accs_train.append(acc)
        losses_val.append(loss_v)
        accs_val.append(acc_v)
        losses_test.append(loss_tt)
        accs_test.append(acc_tt)

        convGradsNorm.append(calc_gradsNorm(gconvNames, Ws))

    return {'trainingLosses': losses_train, 'trainingAccs': accs_train, 'valLosses': losses_val, 'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test, 'convGradsNorm': convGradsNorm}


def eval_gc_prox(model, test_loader, device, gconvNames, mu, Wt):
    model.eval()

    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label) + mu / 2. * _prox_term(model, gconvNames, Wt)
        total_loss += loss.item() * batch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += batch.num_graphs

    return total_loss / ngraphs, acc_sum / ngraphs
