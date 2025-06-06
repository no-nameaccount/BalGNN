import argparse
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import scipy.sparse as sp
import torch

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool).cuda()
    mask[index] = 1
    return mask

def mask_to_index(mask):
    index = torch.where(mask == True)[0].cuda()
    return index


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'{self.info} Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')

    def print_statistics_f(self, final_result,run=None):
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                argmax = r[:, 1].argmax()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)
            
            print(f'{self.info} All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            final_result += (f'Highest Train: {r.mean():.2f} ± {r.std():.2f},')

            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            final_result += (f'  Final Train: {r.mean():.2f} ± {r.std():.2f}, \n')

            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            final_result += (f'   Final Test: {r.mean():.2f} ± {r.std():.2f},')

            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f} \n')
            final_result += (f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')

            return final_result


    def best_result(self, run=None, with_var=False):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            train1 = result[:, 0].max()
            valid  = result[:, 1].max()
            train2 = result[argmax, 0]
            test   = result[argmax, 2]
            return (train1, valid, train2, test)
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                argmax = r[:, 1].argmax()
                train2 = r[argmax, 0].item()
                test = r[argmax, 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            r = best_result[:, 0]
            train1 = r.mean().item()
            train1_var = f'{r.mean():.2f} ± {r.std():.2f}'
            
            r = best_result[:, 1]
            valid = r.mean().item()
            valid_var = f'{r.mean():.2f} ± {r.std():.2f}'

            r = best_result[:, 2]
            train2 = r.mean().item()
            train2_var = f'{r.mean():.2f} ± {r.std():.2f}'

            r = best_result[:, 3]
            test = r.mean().item()
            test_var = f'{r.mean():.2f} ± {r.std():.2f}'

            if with_var:
                return (train1, valid, train2, test, train1_var, valid_var, train2_var, test_var)
            else:
                return (train1, valid, train2, test)


def torch_corr(x):
    mean_x = torch.mean(x, 1)
    xm = x - mean_x.view(-1, 1)
    c = xm.mm(xm.t())

    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    stddev = torch.where(stddev == 0, torch.tensor(1e-6, device=x.device), stddev)

    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    c = torch.clamp(c, -1.0, 1.0)
    return c

def get_pairwise_sim(x):
    try:
        x = x.detach().cpu().numpy()
    except:
        pass

    if sp.issparse(x):
        x = x.todense()
        x = x / (np.sqrt(np.square(x).sum(1))).reshape(-1,1)
        x = sp.csr_matrix(x)
    else:
        x = x / (np.sqrt(np.square(x).sum(1))+1e-10).reshape(-1,1)
    try:
        dis = euclidean_distances(x)
        return 0.5 * (dis.sum(1)/(dis.shape[1]-1)).mean()
    except:
        return -1

def get_random_dimension_pair(x):
    selected_columns = torch.randperm(x.shape[1])[:2] 
    f_i, f_j = x[:, selected_columns[0]], x[:, selected_columns[1]]
    return selected_columns, f_i, f_j

def correlation_coefficient(x, y):
    x = x.view(-1)
    y = y.view(-1)
    
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    cov_xy = torch.mean((x - mean_x) * (y - mean_y))
    std_x = torch.std(x) + 1e-6  
    std_y = torch.std(y) + 1e-6

    corr = cov_xy / (std_x * std_y)
    corr = torch.abs(corr)
    
    return corr