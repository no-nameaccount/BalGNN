"""Based on   https://github.com/hualiu829/DeProp/tree/main"""

import random
import argparse
import time
import itertools

from util import str2bool, Logger
from model import get_model
from train_eval import train, test
from dataset import get_dataset
import torch
import numpy as np

torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser(description='BalGNN')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--normalize_features', type=str2bool, default=True)
    parser.add_argument('--random_splits', type=int, default=5, help='default: fix split')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--fix_split', type=str2bool, default=False)
    parser.add_argument('--alpha', type=int, default = 0.5)
    parser.add_argument('--beta', type=int, default = 0.5)
    parser.add_argument('--missing_rate', type=float, default=100, help='missing_rate.')
    parser.add_argument('--model_type', type=str, default='BalGNN', help='{BalGNN, BalGNN_GAT}')
    parser.add_argument('--log', type=str, default='results.txt')
    args = parser.parse_args()
    return args

def main( ):
    args = parse_args()
    print(f"args:{args}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.random_splits > 0:
        random_split_num = args.random_splits
        print(f'random split {random_split_num} times and each for {args.runs} runs')
    else:
        random_split_num = 1
        print(f'fix split and run {args.runs} times')

    logger = Logger(args.runs * random_split_num)
    total_start = time.perf_counter()

    for split in range(random_split_num):
        dataset, data, split_idx = get_dataset(args, split)
        train_idx = split_idx['train']
        data = data.to(device)
        print("Data:", data)
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()

        model = get_model(args, dataset)
        print(model)

        for run in range(args.runs):
            runs_overall = split * args.runs + run
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_acc_val = 0.2
            t_start = time.perf_counter()
            for epoch in range(1, 1 + args.epochs):
                args.current_epoch = epoch
                loss = train(model, data, train_idx, optimizer, args)
                result, sim, corr, mi = test(model, data, split_idx, args)
                train_acc, valid_acc, test_acc = result
                
                if valid_acc > best_acc_val:
                    best_acc_val = valid_acc
                    print(f"epoch_best:{epoch}")
                    print(f"best_acc_val:{best_acc_val}", f"test_acc: {test_acc}")

                logger.add_result(runs_overall, result)

                if args.log_steps > 0:
                    if epoch % args.log_steps == 0:
                        train_acc, valid_acc, test_acc = result
                        print(f'Split: {split + 1:02d}, '
                              f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_acc:.2f}%, '
                              f'Valid: {100 * valid_acc:.2f}% '
                              f'Test: {100 * test_acc:.2f}%')

            t_end = time.perf_counter()
            duration = t_end - t_start
            if args.log_steps > 0:
                print(print(f'Split: {split + 1:02d}, 'f'Run: {run + 1:02d}'), 'time: ', duration)
                logger.print_statistics(runs_overall)

    total_end = time.perf_counter()
    total_duration = total_end - total_start

    final_result  = ""

    print('total time: ', total_duration)
    final_result = (
        "----------------------------------------"
        f"Dataset: {args.dataset}, Layers: {args.num_layers}, \n"
        f"Total Time: {total_duration:.2f}s,")


    final_result = logger.print_statistics_f(final_result)

    with open(args.log, "a") as log_file:
        log_file.write(final_result + "\n")

if __name__ == "__main__":
    main()




