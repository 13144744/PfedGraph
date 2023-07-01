import pandas
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from torch_geometric.utils import to_networkx
from tqdm import tqdm
import numpy as np
import torch
from collections import Counter
import math
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from sklearn.neighbors import kneighbors_graph


def min_max_Normalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(x):
    x = np.exp(x)
    return x / x.sum()


class Dataset:
    def __init__(self, config, subset):
        data = pandas.read_csv(f"data/{config.dataset}/{subset}_day.csv")
        self.subset = subset
        self.id = list(data['id'])
        self.time = list(data['time'])
        self.event = list(data['event'])
        self.config = config
        self.seq_len = config.seq_len
        self.time_seqs, self.event_seqs = self.generate_sequence()

    def generate_sequence(self):

        pbar = tqdm(total=len(self.id) - self.seq_len + 1)
        time_seqs = []
        event_seqs = []
        cur_end = self.seq_len - 1
        n = 0
        while cur_end < len(self.id):
            n = n + 1
            pbar.update(1)
            cur_start = cur_end - self.seq_len + 1
            if self.id[cur_start] != self.id[cur_end]:
                cur_end += 1
                continue

            subseq = self.time[cur_start:cur_end + 1]
            time_seqs.append(list(subseq))
            event_seqs.append(self.event[cur_start:cur_end + 1])
            cur_end += 1

        time_seqs = time_seqs[:int(len(time_seqs) / self.config.batch_size) * self.config.batch_size]
        event_seqs = event_seqs[:int(len(time_seqs) / self.config.batch_size) * self.config.batch_size]
        return time_seqs, event_seqs

    def generate_adj(self):
        y = []
        dic = {}
        zer = np.zeros((self.config.event_class, self.config.event_class))
        for i in self.event_seqs:
            for j in range(len(i) - 1):
                y.append((i[j], i[j + 1]))

        for i in y:
            dic[i] = dic.get(i, 0) + 1

        for z, y in dic.items():
            zer[z[0], z[1]] = y
        return zer / np.max(zer)

    def __getitem__(self, item):
        return self.time_seqs[item], self.event_seqs[item]

    def __len__(self):
        return len(self.time_seqs)

    @staticmethod
    def to_features(batch):
        times, events = [], []
        for time, event in batch:
            time = np.array([time[0]] + time)
            time = np.diff(time)
            times.append(time)
            events.append(event)
        return torch.FloatTensor(times), torch.LongTensor(events)

    def statistic(self):
        print("TOTAL SEQs:", len(self.time_seqs))
        intervals = np.diff(np.array(self.time))
        for thr in [0.001, 0.01, 0.1, 1, 10, 100]:
            print(f"<{thr} = {np.mean(intervals < thr)}")


def rmse_error(pred, gold):
    return np.sqrt(np.mean((pred - gold) ** 2))


def abs_error(pred, gold):
    return np.mean(np.abs(pred - gold))


def clf_metric(pred, gold, n_class):
    gold_count = Counter(gold)
    pred_count = Counter(pred)
    prec = recall = 0
    pcnt = rcnt = 0
    for i in range(n_class):
        match_count = np.logical_and(pred == gold, pred == i).sum()
        if gold_count[i] != 0:
            prec += match_count / gold_count[i]
            pcnt += 1
        if pred_count[i] != 0:
            recall += match_count / pred_count[i]
            rcnt += 1
    prec /= pcnt
    recall /= rcnt
    print(f"pcnt={pcnt}, rcnt={rcnt}")
    f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1


def read_adj(subset):
    adj = np.loadtxt(f"./data/{subset}/adj.txt")
    adj = get_symmetric_adj(adj)
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def fea_mat(subset, subset2):
    data = pd.read_csv(f"./data/{subset}/{subset2}_day.csv")
    print(f"./data/{subset}/{subset2}_day.csv")
    data_id = list(data['id'])
    data_event = list(data['event'])
    counts = []
    dic = {}
    for i in range(len(data_id)):
        counts.append((data_id[i], data_event[i]))
    for i in counts:
        dic[i] = dic.get(i, 0) + 1
    item = list(dic.items())
    print(set(data_event))
    mat = np.zeros([len(set(data_id)), len(set(data_event))])
    id_list = list(set(data_id))
    event_list = list(set(data_event))
    id_list.sort()
    event_list.sort()

    for i in range(len(set(id_list))):
        a = id_list.index(item[i][0][0])
        b = event_list.index(item[i][0][1])
        mat[a][b] = item[i][1]
    U, Sigma, Vt = np.linalg.svd(mat)
    return Vt


def get_input():
    data = pd.read_csv("/Users/zhouyan/Desktop/hawkes_final/data/atm/train_day.csv")
    x = list(data['id'])
    y = list(data['event'] - 1)
    counts = []
    dic = {}
    for i in range(len(x)):
        counts.append((x[i], y[i]))
    for i in counts:
        dic[i] = dic.get(i, 0) + 1

    item = list(dic.items())
    mat = np.zeros([len(set(x)), len(set(y))])
    id_list = list(set(x))
    event_list = list(set(y))
    id_list.sort()
    event_list.sort()

    for i in range(len(set(id_list))):
        a = id_list.index(item[i][0][0])
        b = event_list.index(item[i][0][1])
        mat[a][b] = item[i][1]

    U, Sigma, Vt = np.linalg.svd(mat)
    User_KNN_Graph = (kneighbors_graph(Vt, 1, mode='connectivity', include_self=True))
    adj = User_KNN_Graph.todense()
    adj = get_symmetric_adj(adj)
    # adj = normalize(adj)
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # U_features = pd.DataFrame([x[0:64] for x in U])

    return adj, np.array(Vt)


def get_symmetric_adj(adj):
    """get symmetric adjacency matrix"""
    adj = coo_matrix(adj)
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


def normalize_adj(adj):
    """D-1/2 *A* D-1/2"""
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    # .flatten()返回一个一位数组
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_adj(adj):
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


# 切分数据
def split_data(graphs, train=None, test=None, shuffle=True, seed=None):
    # y = torch.cat([graph.y for graph in graphs])
    # arr = torch.Tensor.numpy(y)
    # key = np.unique(arr)
    # set_one = []
    # list_one = []
    # for k in key:
    #     v = arr[arr == k].size
    #     if v == 1:
    #         list_one.append(k)
    # # num_graph_labels = len(np.unique(y))
    # # print(list_one)
    # other_sets = []
    # for i in list_one:
    #     for graph in graphs:
    #         if i == graph.y:
    #             set_one.append(graph)
    #             graphs.remove(graph)
    y = torch.cat([graph.y for graph in graphs])
    graphs_tv, graphs_test = train_test_split(graphs, train_size=train, test_size=test, stratify=y, shuffle=shuffle,
                                              random_state=seed)
    # for graph in set_one:
    #     graphs_tv.append(graph)
    return graphs_tv, graphs_test


def split(graphs):
    num_training = int(len(graphs) * 0.8)
    num_val = int(len(graphs) * 0.1)
    num_test = len(graphs) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(graphs, [num_training, num_val, num_test])
    # y = torch.cat([graph.y for graph in training_set])
    # num_graph_labels = len(np.unique(y))

    return training_set, validation_set, test_set


def fingerprint_split(graphs, seed):
    y = torch.cat([graph.y for graph in graphs])

    num_training = int(len(graphs) * 0.8)
    num_val = int(len(graphs) * 0.1)
    num_test = len(graphs) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(graphs, [num_training, num_val, num_test])
    num_graph_labels = len(np.unique(y))

    # num_graph_labels = len(np.unique(y))
    # training_set, vt = split_data(graphs, train=0.8, test=0.2, shuffle=True, seed=seed)
    # validation_set, test_set = split_data(vt, train=0.5, test=0.5, shuffle=True, seed=seed)

    return training_set, validation_set, test_set, num_graph_labels


# 求满度
def get_maxDegree(graphs):
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree

    return maxdegree


# 获取图的标签数量
def get_numGraphLabels(dataset):
    s = set()
    for g in dataset:
        s.add(g.y.item())
    return len(s)


def _get_avg_nodes_edges(graphs):
    numNodes = 0.
    numEdges = 0.
    numGraphs = len(graphs)
    for g in graphs:
        numNodes += g.num_nodes
        numEdges += g.num_edges / 2.  # undirected
    return numNodes / numGraphs, numEdges / numGraphs


def get_stats(df, ds, graphs_train, graphs_val=None, graphs_test=None):
    df.loc[ds, "#graphs_train"] = len(graphs_train)
    avgNodes, avgEdges = _get_avg_nodes_edges(graphs_train)
    df.loc[ds, 'avgNodes_train'] = avgNodes
    df.loc[ds, 'avgEdges_train'] = avgEdges

    if graphs_val:
        df.loc[ds, '#graphs_val'] = len(graphs_val)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_val)
        df.loc[ds, 'avgNodes_val'] = avgNodes
        df.loc[ds, 'avgEdges_val'] = avgEdges

    if graphs_test:
        df.loc[ds, '#graphs_test'] = len(graphs_test)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_test)
        df.loc[ds, 'avgNodes_test'] = avgNodes
        df.loc[ds, 'avgEdges_test'] = avgEdges

    return df
