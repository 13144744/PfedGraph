"""Data utils functions for pre-processing and data loading."""
import math
import random
import time
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    data['adj_train_norm'], data['features'] = process(
        data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data


def process(adj, normalize_adj):
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
        test_edges_false)


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset == 'disease_nc':
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val,
            'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.node[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features


def edge_adj(edge_index, x):
    object_to_idx = {}
    idx_counter = 0
    edge_index = np.array(edge_index.cpu())
    edge_index0, edge_index1 = edge_index[0].tolist(), edge_index[1].tolist()
    edges = []
    for num in range(len(edge_index0)):
        n1, n2 = edge_index0[num], edge_index1[num]
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((n1, n2))
    adj = np.zeros((len(x), len(x)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    a = sp.csr_matrix(adj)
    return sp.csr_matrix(adj)


def reset_batch_adj(databatch):
    edge_index, x = databatch.edge_index, databatch.x
    edge_attr = None
    time1 = time.time()
    adj1 = process(edge_adj(edge_index, x), True)
    time2 = time.time()

    perm = torch.arange(len(x), dtype=torch.long)
    row, col = edge_index
    weights = (torch.cat([x[row], x[col]], dim=1)).sum(dim=-1)
    adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
    adj[row, col] = weights
    time3 = time.time()
    # row, col = edge_index
    # adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
    # adj[row, col] = x
    # adj = process(edge_adj(edge_index, x), True)
    print(time2 - time1)
    print(time3 - time2)
    return adj


def add_adj(dataloader_train):
    batch_adj = []
    for _, dataset in enumerate(dataloader_train):
        # print(dataset.edge_index.shape)
        a = reset_batch_adj(dataset)
        # print("#####")
        # print(a.shape)
        batch_adj.append(a)
    return batch_adj


def get_batch_adj_node(dataloaders, args):
    device = args.device
    train, val, test = [], [], []
    data_train, data_val, data_test = dataloaders['train'], dataloaders['val'], dataloaders['test']

    for batch_num, databatch in enumerate(data_train):
        batch = databatch.batch
        adj = reset_batch_adj(databatch)
        x = databatch.x.to(device)
        y = databatch.y.to(device)
        num_graphs = databatch.num_graphs
        data = {'x': x, 'adj': adj, 'y': y, 'num_graphs': num_graphs, 'batch': batch}
        train.append(data)
    for batch_num, databatch in enumerate(data_val):
        batch = databatch.batch
        adj = reset_batch_adj(databatch)
        x = databatch.x.to(device)
        y = databatch.y.to(device)
        num_graphs = databatch.num_graphs
        data = {'x': x, 'adj': adj, 'y': y, 'num_graphs': num_graphs, 'batch': batch}
        val.append(data)
    for batch_num, databatch in enumerate(data_test):
        batch = databatch.batch
        adj = reset_batch_adj(databatch)
        x = databatch.x.to(device)
        y = databatch.y.to(device)
        num_graphs = databatch.num_graphs
        data = {'x': x, 'adj': adj, 'y': y, 'num_graphs': num_graphs, 'batch': batch}
        test.append(data)
    dataloaders = {'train': train, 'val': val, 'test': test}
    return dataloaders


def mess_up_dataset(unprocessed_data, num_noise_percent, use_edge_attr, dataname):
    # num_noise = int(len(unprocessed_data) * num_noise_percent)
    dataset_1 = []
    y_list, x_list, = [], []
    max_x_size, min_x_size, size_feat, all_x_size, all_edge_size = 0, 9999999, 0, 0, 0
    for dataset in unprocessed_data:
        num_noise = math.ceil(dataset.x.size(0) * num_noise_percent)  # 添加噪音量
        dataset = dataset.to('cpu')
        print(dataset.y)
        actual_labels = torch.unique(dataset.y)  # 获取label
        actual_nodes = np.arange(dataset.x.size(0)).reshape(-1, 1)  # 纵向铺平
        print('actual_nodes:', dataset.x.size(0))
        print('actual_labels:', dataset.y.size())

        real_flags = np.ones(dataset.x.size(0))  # dataset.x数量的1 list
        fake_flags = np.zeros(num_noise)  # noise数量
        flags = np.hstack([real_flags, fake_flags])  # 将两个list合并

        np.random.seed(num_noise)
        torch.manual_seed(num_noise)

        print('> Number of fake data: ', num_noise)

        fake_nodes = np.arange(dataset.x.size(0), dataset.x.size(0) + num_noise)
        size_feat = dataset.x.size(1)
        avg_connect = int(dataset.edge_index.size(1) / dataset.x.size(0))
        # fake data
        fake_labels = torch.tensor(np.random.choice(actual_labels, num_noise).reshape(-1))
        fake_feature = torch.randn(num_noise, size_feat)

        # making fake edges
        real2fake = np.random.choice(fake_nodes, size=(dataset.x.size(0), avg_connect)).reshape(-1)
        fake2real = np.repeat(actual_nodes, avg_connect, axis=-1).reshape(-1)

        np_edge_index = dataset.edge_index.numpy()

        temp_TOP = np.hstack((np_edge_index[0], fake2real))
        idx_sorting = np.argsort(temp_TOP)
        TOP = np.sort(temp_TOP)
        temp_bottom = np.hstack([np_edge_index[1], real2fake])
        BOTTOM = temp_bottom[idx_sorting]

        REAL_add = np.vstack([TOP, BOTTOM])
        FAKE_add = np.vstack([real2fake, fake2real])

        # all-together
        dataset.edge_index = torch.tensor(np.hstack([REAL_add, FAKE_add]))
        dataset.x = torch.cat([dataset.x, fake_feature], dim=0)
        # dataset.y = torch.cat([dataset.y, fake_labels], dim=-1)
        dataset.flags = torch.tensor(flags)
        print('fake_nodes:', dataset.x.size(0))
        print('fake_labels:', dataset.y.size())
        dataset_1.append(dataset)
    return dataset_1


def sd_matrixing(state_dic):
    """
    Turn state dic into a vector
    :param state_dic:
    :return:
    """
    keys = []
    param_vector = None
    for key, param in state_dic.items():
        keys.append(key)
        if param_vector is None:
            param_vector = param.clone().detach().flatten().cpu()
        else:
            if len(list(param.size())) == 0:
                param_vector = torch.cat((param_vector, param.clone().detach().view(1).cpu().type(torch.float32)), 0)
            else:
                param_vector = torch.cat((param_vector, param.clone().detach().flatten().cpu()), 0)
    return param_vector


def list_matrixing(state_dic):
    """
    Turn a list into a vector
    :param state_dic:
    :return:
    """
    keys = []
    param_vector = None
    for key, param in state_dic.items():
        # keys.append(key)
        if param_vector is None:
            param_vector = param.clone().detach().flatten()
        else:
            if len(list(param.size())) == 0:
                param_vector = torch.cat((param_vector, param.clone().detach().view(1).type(torch.float32)), 0)
            else:
                param_vector = torch.cat((param_vector, param.clone().detach().flatten()), 0)
    return param_vector
