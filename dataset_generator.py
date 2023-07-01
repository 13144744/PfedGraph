import math

import numpy as np
import random
import torch
# import spacy
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split


# from torchtext.datasets import AG_NEWS
# from torchtext.legacy.data import Field, Batch
# from torchtext.legacy.datasets import Multi30k
# from utils.miniboone_solver import load_data_idx, MiniBooNEDatasetSolver
# from utils.nlp_supporter import collate_batch


def split_list_n_list(origin_list, n):
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1
    for i in range(0, n):
        yield origin_list[i * cnt: (i + 1) * cnt]


class DatasetSplit(Dataset):
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = [int(i) for i in idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        image, label = self.dataset[self.idx[item]]
        return torch.tensor(image), torch.tensor(label)


class DatasetSplitNLP(Dataset):
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = [int(i) for i in idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        return self.dataset[self.idx[item]]


def to_map_style_dataset(iter_data):
    # Inner class to convert iterable-style to map-style dataset
    class _MapStyleDataset(Dataset):

        def __init__(self, iter_data):
            # TODO Avoid list issue #1296
            self._data = list(iter_data)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

    return _MapStyleDataset(iter_data)


def get_train_loader(arg, dataset):
    train_loader = []

    dataset_size = len(dataset)
    worker_dataset_size_not_last = int(math.floor(dataset_size / arg.world_size))
    worker_dataset_size_last = dataset_size - (arg.world_size - 1) * worker_dataset_size_not_last
    # 序列之和等于dataset_size
    list = [worker_dataset_size_not_last for _ in range(arg.world_size - 1)]
    list.append(worker_dataset_size_last)
    subset = random_split(dataset, list)
    for i in range(len(subset)):
        # if arg.dataset == 'agnews':
        #     train_loader.append(torch.utils.data.DataLoader(subset[i], batch_size=arg.batch_size, shuffle=True, collate_fn=collate_batch))
        # else:
        train_loader.append(torch.utils.data.DataLoader(subset[i], batch_size=arg.batch_size, shuffle=True))
    return train_loader, list


def non_iid_one_class(train_dataset, world_size):
    current_training_dataset_idx = 0
    idx_train = {i: np.array([]) for i in range(world_size)}
    # idx_list保存10个list，分别存放 0-10 不同label的index
    idx_list = [[] for _ in range(world_size)]
    for data in train_dataset:
        # data[1] 输出显示 mnist 的 label
        idx_list[data[1]].append(current_training_dataset_idx)
        current_training_dataset_idx += 1
    for i in range(world_size):
        random.shuffle(idx_list[i])
        idx_train[i] = np.array(idx_list[i])
    return idx_train


def non_iid_x_class(train_dataset, world_size, x):
    # 如果是 MNIST或 CIFAR10 保证 x 小于 10
    current_training_dataset_idx = 0
    idx_train = {i: np.array([]) for i in range(world_size)}
    # 为了方便起见 node0->{0,1}; node1->{1,2};....node9->{9,0}
    idx_list = [[] for _ in range(world_size)]
    for data in train_dataset:
        # data[1] 输出显示 mnist 的 label
        idx_list[data[1]].append(current_training_dataset_idx)
        current_training_dataset_idx += 1
    # shuffle
    for i in range(len(idx_list)):
        random.shuffle(idx_list[i])
    training_label_each_node = [[] for _ in range(world_size)]
    # 每个 node 的数据集应该包含那些labels 保存在 training_label_each_node
    for i in range(world_size):
        for j in range(x):
            training_label_each_node[i].append((i + j) % world_size)

    generator_list = []
    # 对已经分类 shuffle 过的 idx_list 进行切分
    for l in idx_list:
        generator_list.append(split_list_n_list(l, x))

    for i in range(world_size):
        list = []
        for j in range(len(training_label_each_node[i])):
            # training_label_each_node[i][j] 表示 label
            list.extend(next(generator_list[training_label_each_node[i][j]]))
        random.shuffle(list)
        idx_train[i] = np.array(list)
    return idx_train


def sample_dirichlet_train_data_cv(train_dataset, world_size, param):
    classes = {}  # {label1:[idx....], label:[idx...]}
    for idx, data in enumerate(train_dataset):
        _, label = data
        if label in classes:
            classes[label].append(idx)
        else:
            classes[label] = [idx]
    num_classes = len(classes.keys())  # the num of labels
    classes_size = len(classes[0])
    dict_users_train = {i: {'data': np.array([]), 'label': set()} for i in range(world_size)}

    for n in range(num_classes):
        random.shuffle(classes[n])
        sampled_probalities = classes_size * np.random.dirichlet(np.array((world_size * [param])))
        for user in range(world_size):
            num_imgs = int(round(sampled_probalities[user]))
            sampled_list = classes[n][:min(len(classes[n]), num_imgs)]
            dict_users_train[user]['data'] = np.concatenate((dict_users_train[user]['data'], np.array(sampled_list)),
                                                            axis=0)
            if num_imgs > 0:
                dict_users_train[user]['label'].add(n)
            classes[n] = classes[n][min(len(classes[n]), num_imgs):]
        # shuffle data
    for user in range(world_size):
        random.shuffle(dict_users_train[user]['data'])

    for i in range(world_size):
        classes = {label: {'num': 0} for label in range(num_classes)}
        print(f'client: {i} / data size: {len(dict_users_train[i]["data"])}')
        for idx in list(dict_users_train[i]["data"]):
            _, label = train_dataset[idx.astype('int64')]
            classes[label]['num'] += 1
        print(f'client: {i}, {classes}')
    return dict_users_train


def sample_dirichlet_train_data_nlp(train_set, world_size, dirchlet):
    classes = {}  # {label1:[idx....], label:[idx...]}
    for idx, (label, text) in enumerate(train_set):
        if label in classes:
            classes[label].append(idx)
        else:
            classes[label] = [idx]
    num_classes = len(classes.keys())  # the num of labels
    classes_size = 30000  # default dataset: AG News
    dict_users_train = {i: {'data': np.array([]), 'label': set()} for i in range(world_size)}

    for n in range(num_classes):
        random.shuffle(classes[n + 1])
        sampled_probalities = classes_size * np.random.dirichlet(np.array((world_size * [dirchlet])))
        for user in range(world_size):
            num_imgs = int(round(sampled_probalities[user]))
            sampled_list = classes[n + 1][:min(len(classes[n + 1]), num_imgs)]
            dict_users_train[user]['data'] = np.concatenate((dict_users_train[user]['data'], np.array(sampled_list)),
                                                            axis=0)
            if num_imgs > 0:
                dict_users_train[user]['label'].add(n + 1)
            classes[n + 1] = classes[n + 1][min(len(classes[n + 1]), num_imgs):]
        # shuffle data
    for user in range(world_size):
        random.shuffle(dict_users_train[user]['data'])

    for i in range(10):
        classes = {label + 1: {'num': 0} for label in range(num_classes)}
        print(f'client: {i} / data size: {len(dict_users_train[i]["data"])}')
        for idx in list(dict_users_train[i]["data"]):
            (label, text) = train_set[idx.astype('int64')]
            classes[label]['num'] += 1
        print(f'client: {i}, {classes}')
    return dict_users_train


def custom_noniid_dataset(args, transform):
    train_set, test_set, idx_train = None, None, None
    train_loader, data_size_partition = [], []
    data_dir = './data'
    if args.dataset == 'mnist':
        train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    elif args.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    # elif args.dataset == 'agnews':
    #     train_iter, test_iter = AG_NEWS(root=data_dir)
    #     train_set = to_map_style_dataset(train_iter)
    #     test_set = to_map_style_dataset(test_iter)
    # elif args.dataset == 'shakespeare':
    #     train_set, test_set = Shakespare(train=True), Shakespare(train=False)
    #     dict_lists = train_set.get_client_dic()
    #     rmd_idx_list = np.random.choice(range(len(dict_lists)), args.world_size, replace=False)  # 随机选择数据切片
    #     for idx in rmd_idx_list:
    #         train_loader.append(
    #             torch.utils.data.DataLoader(DatasetSplit(train_set, dict_lists[idx]), batch_size=args.batch_size,
    #                                         shuffle=True))
    #         print(len(dict_lists[idx]))
    #         data_size_partition.append(len(dict_lists[idx]))
    #     test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    #     return train_loader, test_loader, data_size_partition
    if args.dataset == 'mnist' or args.dataset == 'cifar10':
        if args.x == 1:
            print(f'CV x: {args.x}')
            idx_train = non_iid_one_class(train_set, args.world_size)
        elif args.x > 1:
            print(f'CV x: {args.x}')
            idx_train = non_iid_x_class(train_set, args.world_size, args.x)
        else:  # x == 0
            print(f'CV Tasks Dirichlet Sampling')
            idx_train = sample_dirichlet_train_data_cv(train_set, args.world_size, args.dirichlet)
            np.save(f'npy/idx_train_{args.dataset}_{args.dirichlet}_{args.world_size}.npy', idx_train,
                    allow_pickle=True)
            # idx_train = np.load(f'npy/idx_train_{args.dataset}_{args.dirichlet}_{args.world_size}.npy', allow_pickle=True).item()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    if idx_train is not None:
        for value in idx_train.values():
            if args.dataset == 'mnist' or args.dataset == 'cifar10':
                if args.x != 0:
                    idx_list = list(value)
                else:
                    idx_list = list(value['data'])
                if args.dataset == 'mnist':
                    train_loader.append(
                        torch.utils.data.DataLoader(DatasetSplit(train_set, idx_list), batch_size=args.batch_size,
                                                    shuffle=True))
                    data_size_partition.append(len(value))
                elif args.dataset == 'cifar10':
                    train_loader.append(
                        torch.utils.data.DataLoader(DatasetSplit(train_set, idx_list), batch_size=args.batch_size,
                                                    shuffle=True))
                    data_size_partition.append(len(value))
    return train_loader, test_loader, 0


# def miniboone_dataset(args):
#     data, train_idx_list, test_idx = load_data_idx('./data/MiniBooNE/MiniBooNE.npy', 0.1, args.world_size)
#     train_loader_list, data_size_partition_list = [], []
#     test_loader = torch.utils.data.DataLoader(MiniBooNEDatasetSolver(data, test_idx), batch_size=args.batch_size,
#                                               shuffle=False)
#     for idx_list in train_idx_list:
#         train_loader_list.append(
#             torch.utils.data.DataLoader(MiniBooNEDatasetSolver(data, idx_list), batch_size=args.batch_size,
#                                         shuffle=True))
#         data_size_partition_list.append(len(idx_list))
#     return train_loader_list, test_loader, data_size_partition_list

# def multi30k_dataset(args):
#     data_dir = './data'
#     # download the spacy model first! Initialize tokenlizer
#     # python -m spacy download de_core_news_sm
#     # python -m spacy download en_core_web_sm
#     spacy_de = spacy.load('de_core_news_sm')
#     spacy_en = spacy.load('en_core_web_sm')
#
#     # tokenize text from a string into a list of strings in lambda
#     # <sos> start of sequence
#     # <eos> end of sequence
#     source = Field(tokenize=lambda text: [tok.text for tok in spacy_de.tokenizer(text)],
#                    init_token='<sos>', eos_token='<eos>', lower=True)
#     target = Field(tokenize=lambda text: [tok.text for tok in spacy_en.tokenizer(text)],
#                    init_token='<sos>', eos_token='<eos>', lower=True)
#     # Multi30k数据集 一个包含约30000个平行的英语、德语、法语句子的数据集 每个句子包含约12个单词
#     train_data, valid_data, test_data = Multi30k.splits(root=data_dir, exts=('.de', '.en'), fields=(source, target))
#     # torchtext.datasets.AG_NEWS
#     # 构建词表 即给每个单词编码 用数字表示每个单词
#     # 源语言和目标语言的词表时不同的 而且词表只从训练集中构建 而不是验证集或测试集
#     source.build_vocab(train_data, min_freq=2)  # min_freq=2 最小词频 当一个单词在数据集中出现的次数小于2时会被转换到<unk>字符
#     target.build_vocab(train_data, min_freq=2)
#     input_dim, output_dim = len(source.vocab), len(target.vocab)
#
#     def torchtext_collate(data):
#         b = Batch(data, train_data)
#         return {'src': b.src, 'trg': b.trg}
#
#     for i, example in enumerate(train_data.examples):
#         print(i)
#         print(vars(example))
#
#     sampler = DistributedSampler(train_data, num_replicas=args.world_size, rank=0)
#     train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=torchtext_collate, sampler=sampler,
#                               shuffle=False)
#     valid_loader = DataLoader(valid_data, batch_size=args.batch_size, collate_fn=torchtext_collate, sampler=sampler,
#                               shuffle=False)
#     test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=torchtext_collate, sampler=sampler,
#                              shuffle=False)
#     return train_loader, valid_loader, test_loader, input_dim, output_dim

def data_distributor(args):
    transform = None
    mnist_lr_transform = torchvision.transforms.ToTensor()
    mnist_lenet_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    cifar10_lenet_transform = torchvision.transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    cifar10_resnet_transform = torchvision.transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    miniboone_lenet_transform = torchvision.transforms.ToTensor()
    if args.dataset == 'mnist':
        if args.model == 'LR':
            transform = mnist_lr_transform
            return custom_noniid_dataset(args, transform)
        elif args.model == 'lenet':
            transform = mnist_lenet_transform
            return custom_noniid_dataset(args, transform)
    elif args.dataset == 'cifar10':
        if args.model == 'resnet':
            transform = cifar10_resnet_transform
            return custom_noniid_dataset(args, transform)
        elif args.model == 'lenet':
            transform = cifar10_lenet_transform
            return custom_noniid_dataset(args, transform)
    # elif args.dataset == 'miniboone':
    #     transform = miniboone_lenet_transform
    #     return miniboone_dataset(args)
    # elif args.dataset == '30k':
    #     return multi30k_dataset(args)
    # elif args.dataset == 'agnews':
    #     return custom_noniid_dataset(args, None)
    # elif args.dataset == 'shakespeare':
    #     return custom_noniid_dataset(args, None)
