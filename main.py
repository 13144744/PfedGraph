import copy
from datetime import datetime

# import geoopt

from utils import setup
from argparse import ArgumentParser
import os
import torch

from training import *


def process_HGCNAggregate(clients, server, model, name):
    print("Running FedAvg With {model}...")
    filename = name
    frame1, frame2 = run_HGCNAggregate(args, clients, server, args.num_rounds, args.local_epoch, samp=None)
    # print(frame)
    outfile = os.path.join(args.outpath + '/' + str(args.dirichlet) + '/' + args.dataset,
                           f'{args.dataset}_HGCNFLwithLoss_{args.gc_epoch}_{model}_client{args.num_clients}_{filename}_{args.num_rounds}_{args.name}.csv')
    frame1.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_time", type=int, default="1")
    parser.add_argument("--name", help='',
                        type=str, default='{:%Y_%m_%d_%H_%M_%S_%f}'.format(datetime.now()))
    parser.add_argument("--data_type", help='graph_base',
                        type=str, default='not_graph')
    parser.add_argument("--datapath", help='data path',
                        type=str, default="./data")
    parser.add_argument("--outpath", default='./log_adddirichlet', type=str)
    parser.add_argument("--dataset",
                        help='data set:PROTEINS/NCI1/IMDB-MULTI/DD',
                        type=str, default='PROTEINS')

    parser.add_argument('--num_rounds', type=int, default=500,
                        help='number of rounds to simulate;')
    # clients
    parser.add_argument('--num_clients', help='number of clients',
                        type=int, default=10)
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--model_clients', help='data set:GCN/GAT/GraphSAGE',
                        type=str, default="GCN")

    parser.add_argument("--hid_dim", help='dimension of hidden layer neurons',
                        type=int, default=64)
    parser.add_argument("--num_layers", help='Model layer',
                        type=int, default=3)
    parser.add_argument("--c", help="Curvature of hyperbolic space",
                        type=float, default=1)

    parser.add_argument("--dropout", help='drop out',
                        type=float, default=0.1)
    parser.add_argument("--batch_size", help='batch size',
                        type=int, default=64)
    parser.add_argument("--wd", help='weight decay',
                        type=float, default=5e-4)
    parser.add_argument("--lr", help='learning rate',
                        type=int, default=0.01)
    parser.add_argument("--SEED", help='seed', type=int, default=5678)
    parser.add_argument("--manifold", type=str, default="PoincareBall")
    parser.add_argument("--use_bias", type=bool, default=False)
    parser.add_argument("--adam_betas", help='', type=str, default="0.9,0.999")
    parser.add_argument("--act", help='', default='relu')
    parser.add_argument("--task", help='', default='relu')
    parser.add_argument("--bias", help='', default=1)
    parser.add_argument("--use-att", help='', default=0)
    parser.add_argument("--pos_weight", help='',
                        type=int, default=0)
    parser.add_argument("--sgd", help='',
                        action='store_true')
    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
    parser.add_argument('--dirichlet',
                        help='0: We dont use this method, 0.1: The noniid index is 0.1, 1: The noniid index is 1, 10: The noniid index is 10',
                        type=float, default=20)

    parser.add_argument('--gc_lr', type=float, default=0.001)
    parser.add_argument('--gc_epoch', type=float, default=50)
    parser.add_argument('--gc_ratio', type=float, default=1)
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()

    print(args.device)

    seed_dataSplit = 123

    for i in range(10):
        time = i + 1
        splitedData, df_stats = setup.prepareData_oneDS(
            datapath=args.datapath,
            data=args.dataset,
            num_client=args.num_clients,
            batchSize=args.batch_size,
            convert_x=args.convert_x,
            seed=seed_dataSplit,
            overlap=args.overlap,
            dirichlet=args.dirichlet
        )

        print("Done")
        print("device" + str(args.device))
        print("non-iid" + str(args.dirichlet))
        id_process = os.getpid()

        init_clients, init_server = setup.setup_devices(splitedData, args, args.model_clients)
        process_HGCNAggregate(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server),
                              model=args.model_clients, name=time)
