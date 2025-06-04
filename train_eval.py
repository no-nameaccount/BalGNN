import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator


def train(model, data, train_idx, optimizer, args):
    model.train()
    optimizer.zero_grad()
    out = model(x=data.x, adj_t=data.adj_t, y = data.y)[train_idx]

    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  

    loss = args.beta * F.nll_loss(out, y)
    loss += model.loss_corr 
    loss.backward()
    optimizer.step()

    if model.loss_corr!=0:    
        model.loss_corr = 0

    return loss.item()



@torch.no_grad()
def test(model, data, split_idx, args):
    model.eval()
    with torch.no_grad():
        out, sim, corr, mi = model(x=data.x, adj_t=data.adj_t, y = data.y, test_true=True) 

    y_pred = out.argmax(dim=-1, keepdim=True)
    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1) # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return (train_acc, valid_acc, test_acc), sim, corr, mi



