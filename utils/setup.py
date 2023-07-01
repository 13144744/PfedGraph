import random
from random import choices

from sklearn.utils import shuffle

from models.client_model import *
from models.client import *
from models.server import *
from models.server_model import *
from utils.util import *
import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree


def reset_batch_adj(data):
    print(data['train'].dataset)
    print(data['train'].dataset[0])
    pass


def _randChunk(graphs, num_client, overlap, seed=None, dirichlet=0):
    random.seed(seed)
    np.random.seed(seed)
    if dirichlet == 10:
        totalNum = len(graphs)
        minSize = min(50, int(totalNum / num_client))
        graphs_chunks = []
        if True:
            for i in range(num_client):
                graphs_chunks.append(graphs)
        return graphs_chunks
    else:
        minSize = int(min(10, int(len(graphs) / num_client)) / 2)
        begin_graph = graphs[:num_client * minSize]
        mid = graphs[num_client * minSize:]
        mid_graph = mid[:len(mid) - num_client * minSize]
        end_graph = mid[len(mid) - num_client * minSize:]
        graphs_chunks = []
        if not overlap:
            for i in range(num_client):
                graphs_chunks.append(begin_graph[i * minSize:(i + 1) * minSize])

            idx = np.random.randint(low=0, high=num_client, size=1)[0]
            for x in range(idx):
                random.shuffle(mid_graph)
            rd = np.random.dirichlet(np.array((num_client * [dirichlet])))
            totalNum = len(mid_graph) * rd
            used_chunks = 0
            for i in range(num_client):
                num_chunks = int(totalNum[i])
                begin = used_chunks
                end = used_chunks + num_chunks
                for g in mid_graph[begin:end]:
                    graphs_chunks[i].append(g)
                used_chunks = end
            for g in mid_graph[used_chunks:]:
                idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
                graphs_chunks[idx_chunk].append(g)

            for i in range(num_client):
                for g in end_graph[i * minSize:(i + 1) * minSize]:
                    graphs_chunks[i].append(g)
        else:
            sizes = np.random.randint(low=50, high=150, size=num_client)
            for s in sizes:
                graphs_chunks.append(choices(graphs, k=s))
        return graphs_chunks


def prepareData_oneDS(datapath, data, num_client, batchSize, convert_x=False, seed=None, overlap=False, dirichlet=0):
    use_node_attr, use_edge_attr = False, False
    frame_data = pd.DataFrame()
    if data == "COLLAB":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=use_node_attr, use_edge_attr=use_edge_attr,
                              pre_transform=OneHotDegree(491, cat=False))
    elif data == "IMDB-BINARY":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=use_node_attr, use_edge_attr=use_edge_attr,
                              pre_transform=OneHotDegree(135, cat=False))
    elif data == "IMDB-MULTI":
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=use_node_attr, use_edge_attr=use_edge_attr,
                              pre_transform=OneHotDegree(88, cat=False))
    elif data == "Letter-low":
        use_node_attr, use_edge_attr = True, True
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=use_node_attr, use_edge_attr=use_edge_attr)
    elif data == "Fingerprint":
        use_node_attr, use_edge_attr = True, False
        tudataset = TUDataset(f"{datapath}/TUDataset", "Fingerprint", use_node_attr=use_node_attr,
                              use_edge_attr=use_edge_attr)
    else:
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=use_node_attr, use_edge_attr=use_edge_attr)
        if convert_x:
            maxdegree = get_maxDegree(tudataset)
            tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
    graphs = []

    for x in tudataset:
        graphs.append(x)

    print("  **", data, len(graphs))
    graphs_chunks = _randChunk(graphs, num_client, overlap, seed=seed, dirichlet=dirichlet)
    splitedData = {}
    df = pd.DataFrame()
    num_node_features = graphs[0].num_node_features
    num_graph_labels = get_numGraphLabels(graphs)
    j = 0
    for idx, chunks in enumerate(graphs_chunks):
        ds = f'{idx}-{data}'
        ds_tvt = chunks
        for i in ds_tvt:
            label = i.y.item()
            frame_data.loc[j, 'label'] = label
            frame_data.loc[j, 'idx'] = idx
            j = j + 1
        graphs_train, graphs_val, graphs_test = split(ds_tvt)

        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)

        splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                           num_node_features, num_graph_labels, len(graphs_train))
        df = get_stats(df, ds, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)
    outfile = os.path.join("./figs",
                           f'data_distribute_IMDBM.csv')
    frame_data.to_csv(outfile)
    return splitedData, df


def prepareData_fingerprint(datapath, data, num_client, batchSize, convert_x=False, seed=None, overlap=False,
                            dirichlet=0):
    use_node_attr, use_edge_attr = False, False
    if data == "Fingerprint":
        use_node_attr = True
        tudataset = TUDataset(f"{datapath}/TUDataset", "Fingerprint", use_node_attr=use_node_attr,
                              use_edge_attr=use_edge_attr)
    else:
        tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=use_node_attr, use_edge_attr=use_edge_attr)
        if convert_x:
            maxdegree = get_maxDegree(tudataset)
            tudataset = TUDataset(f"{datapath}/TUDataset", data, transform=OneHotDegree(maxdegree, cat=False))
    graphs = []

    for x in tudataset:
        graphs.append(x)
    print("  **", data, len(graphs))
    graphs_chunks = _randChunk(graphs, num_client, overlap, seed=seed, dirichlet=dirichlet)
    splitedData = {}
    df = pd.DataFrame()
    num_node_features = graphs[0].num_node_features
    num_graph_labels = get_numGraphLabels(graphs)
    for idx, chunks in enumerate(graphs_chunks):
        ds = f'{idx}-{data}'
        ds_tvt = chunks
        graphs_train, graphs_valtest = split_data(ds_tvt, train=0.8, test=0.2, shuffle=True, seed=seed)
        graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)

        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)

        splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                           num_node_features, num_graph_labels, len(graphs_train))
        df = get_stats(df, ds, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)

    return splitedData, df


def setup_devices(splitedData, args, use_model):
    idx_clients = {}
    clients = []
    # "GCN" "GAT" "GraphSAGE"
    if use_model == "GCN":
        for idx, ds in enumerate(splitedData.keys()):
            print(ds + " preparing")
            idx_clients[idx] = ds
            dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
            print("num_node_features=" + str(num_node_features) + " num_graph_labels=" + str(num_graph_labels))
            cmodel = client_GCN(args, num_node_features, args.hid_dim, num_graph_labels)
            optimizer = torch.optim.Adam(cmodel.parameters(), lr=args.lr, weight_decay=args.wd)
            clients.append(Client_Net(cmodel, idx, ds, train_size, dataloaders, optimizer, args))
        smodel = server_GCN(args, num_node_features, args.hid_dim, num_graph_labels)
        server = Server_Net(smodel, args.device, args)
    elif use_model == "GAT":
        for idx, ds in enumerate(splitedData.keys()):
            print(ds + " preparing")
            idx_clients[idx] = ds
            dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
            print("num_node_features=" + str(num_node_features) + " num_graph_labels=" + str(num_graph_labels))
            cmodel = GAT(num_node_features, args.hid_dim, num_graph_labels, args.num_layers, args.dropout, )
            optimizer = torch.optim.Adam(cmodel.parameters(), lr=args.lr, weight_decay=args.wd)
            clients.append(Client_Net(cmodel, idx, ds, train_size, dataloaders, optimizer, args))
        smodel = serverGAT(num_node_features, args.hid_dim, num_graph_labels, args.num_layers, args.dropout, )
        server = Server_Net(smodel, args.device, args)
    elif use_model == "GraphSAGE":
        for idx, ds in enumerate(splitedData.keys()):
            print(ds + " preparing")
            idx_clients[idx] = ds
            dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
            print("num_node_features=" + str(num_node_features) + " num_graph_labels=" + str(num_graph_labels))
            cmodel = GraphSAGE(num_node_features, args.hid_dim, num_graph_labels, args.num_layers, args.dropout, )
            optimizer = torch.optim.Adam(cmodel.parameters(), lr=args.lr, weight_decay=args.wd)
            clients.append(Client_Net(cmodel, idx, ds, train_size, dataloaders, optimizer, args))
        smodel = serverGraphSAGE(num_node_features, args.hid_dim, num_graph_labels, args.num_layers, args.dropout, )
        server = Server_Net(smodel, args.device, args)
    return clients, server
