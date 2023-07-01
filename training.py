import math

import pandas as pd
import numpy as np
import time
import copy

import torch
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from torch import relu, sigmoid
from torch.distributed import reduce

def run_HGCNAggregate(args, clients, server, COMMUNICATION_ROUNDS, local_epoch=10, samp=None, frac=1.0):
    frame1, frame2 = pd.DataFrame(), pd.DataFrame()
    client_number = len(clients)
    for client in clients:
        client.download_from_server(server)
    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    # start
    avg_metrix = None
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        start = time.perf_counter()
        loss_locals, acc_locals, client_num = [], [], []
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        avg_train_loss = 0.0
        for client in clients:
            train_loss, train_acc = client.local_HGCNFL_train(local_epoch, avg_metrix)
            avg_train_loss = avg_train_loss + train_loss
            loss_locals.append(copy.deepcopy(train_loss))
            acc_locals.append(copy.deepcopy(train_acc))
            client_num.append(client.train_size)

        loss_train = avg_train_loss / client_number
        new_selected_clients, avg_metrix = server.hgcn_aggregate_weights(args, "HGCN", clients)  # server做聚合
        for client in clients:
            for k in client.W:
                client.W[k].data = new_selected_clients[client.id][k].clone().to(args.device)

        avg_loss = 0.0
        avg_acc = 0.0
        i = 0
        for client in clients:
            i = i + 1
            loss_t, acc_t = client.evaluate()
            frame2.loc[client.name, 'test_acc'] = acc_t
            avg_loss = avg_loss + loss_t
            avg_acc = avg_acc + acc_t
        avg_loss = avg_loss / client_number
        avg_acc = avg_acc / client_number
        costtime = time.perf_counter() - start
        frame1.loc[str(c_round), 'avg_loss'] = avg_loss
        frame1.loc[str(c_round), 'avg_acc'] = avg_acc
        frame1.loc[str(c_round), 'time'] = costtime

        print('Iteration: {:04d}'.format(c_round),
              'loss_train: {:.6f}'.format(loss_train),
              'loss_test: {:.6f}'.format(avg_loss),
              'acc_test: {:.6f}'.format(avg_acc),
              'time: {:.6f}'.format(costtime))

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame2.style.apply(highlight_max).data
    print(fs)
    return frame1, frame2
