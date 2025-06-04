"""Based on   https://github.com/hualiu829/DeProp/tree/main"""

import torch
import os
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB
from util import index_to_mask, mask_to_index
import numpy as np

def get_dataset(args, split, sparse=True, **kwargs):

    if sparse:
        transform = T.ToSparseTensor()
    else:
        transform=None

    ## fix random seed for data split
    seeds_init = [12232231, 12232432, 4665565, 45543345, 454543543, 45345234, 54552234, 234235425, 909099343]
    seeds = []
    for i in range(1, 20):
        seeds = seeds + [a*i for a in seeds_init]
    seed = seeds[split]
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)

    if args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
        dataset = Planetoid(root=path, name=args.dataset)
    elif args.dataset == "texas" or args.dataset == "cornell" or args.dataset == "wisconsin":
        dataset = WebKB(root=path, name=args.dataset)
    elif args.dataset == "chameleon" or args.dataset == "squirrel":
        dataset = WikipediaNetwork(root=path, name=args.dataset)
    else:
        print(f"Wrong dataset")

    dataset.transform = get_transform(args.normalize_features, transform)
    data = dataset[0]
    if args.random_splits > 0:
        data = random_planetoid_splits(data, num_classes=dataset.num_classes, seed=seed, args=args)
        print(f'random split {args.dataset} split {split}')

    split_idx = {}
    split_idx['train'] = mask_to_index(data.train_mask)
    split_idx['valid'] = mask_to_index(data.val_mask)
    split_idx['test']  = mask_to_index(data.test_mask)

    return dataset, data, split_idx



def get_transform(normalize_features, transform):
    if transform is not None and normalize_features:
        transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        transform = T.NormalizeFeatures()
    elif transform is not None:
        transform = transform
    return transform


def random_planetoid_splits(data, num_classes, seed, args):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)

    if args.fix_split:
        print(f"split: 20, 500, 1000")
        train_index = torch.cat([i[:20] for i in indices], dim=0)
        rest_index = torch.cat([i[20:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)
    else:
        # import ipdb;ipdb.set_trace()
        print(f"split: 60, 20, 20")
        train_index = torch.cat([i[:round(0.6 * len(i))] for i in indices], dim=0)
        valid_index = torch.cat([i[round(0.6 * len(i)): round(0.8 * len(i))] for i in indices], dim=0)
        test_index = torch.cat([i[round(0.8 * len(i)):] for i in indices], dim=0)

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(valid_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data

