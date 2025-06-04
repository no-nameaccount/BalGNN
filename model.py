import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn import GATConv
from util import  *
from infomax import Infomax
from sklearn.feature_selection import mutual_info_classif


def MI_measure(x, y):
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy().astype(int).ravel()
    
    mi_scores = mutual_info_classif(x_np, y_np, discrete_features='auto')
    return sum(mi_scores)


def matrix_norm(x, epsilon=1e-6):
    mean = x.mean()
    std = x.std() + epsilon
    return (x - mean) / std

class BalGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_channels, dropout, args):
        super(BalGNN, self).__init__()
        self.alpha = args.alpha
        self.num_layers = args.num_layers
        self.dropout = dropout
        
        # Convolutional layers
        self.conv_in = GCNConv(in_dim, hidden_dim)
        self.convs = torch.nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(self.num_layers - 2)
        ])
        self.conv_out = GCNConv(hidden_dim, out_channels)

        # Infomax and correlation loss
        self.infomax = Infomax(1, 1)
        self.loss_corr = 0

    def reset_parameters(self):
        self.conv_in.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv_out.reset_parameters()

    def apply_infomax_loss(self, x, y, i):
        if i % 2 == 0: 
            for _ in range(2): 
                _, f_i, f_j = get_random_dimension_pair(x)
                f_i, f_j = f_i.unsqueeze(1).float(), f_j.unsqueeze(1).float()
                cc3 = correlation_coefficient(f_j, f_i)
                self.loss_corr += (self.infomax.get_loss(f_i, y) + 
                                   self.infomax.get_loss(f_j, y) + 
                                   self.alpha * cc3)

    def forward(self, x, adj_t, y, train_adj=None, test_true=False):
        if not isinstance(y, bool):
            y = y.unsqueeze(1).float() 

        # Input layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_in(x, adj_t)
        x = matrix_norm(x) 
        x = F.relu(x)
        x_init = x

        self.apply_infomax_loss(x, y, i=0)

        # Hidden layers
        for i, conv in enumerate(self.convs, start=1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, adj_t)
            x = matrix_norm(x)
            x = F.relu(x)
            self.apply_infomax_loss(x, y, i)

        # Initial feature flow
        x = x + x_init

        mi = MI_measure(x, y)

        # Output layer
        x = self.conv_out(x, adj_t)

        if test_true:
            corr = torch_corr(x.t())
            corr = torch.triu(corr, 1).abs()
            n = corr.shape[0]
            corr = corr.sum().item() / (n * (n - 1) / 2)
            sim = get_pairwise_sim(x)
            return F.log_softmax(x, dim=1), sim, corr, mi

        return F.log_softmax(x, dim=1)

class BalGNN_GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_channels, dropout, args):
        super(BalGNN_GAT, self).__init__()
        self.alpha = args.alpha
        self.num_layers = args.num_layers
        self.dropout = dropout
        
        # Convolutional layers
        self.conv_in = GATConv(in_dim, hidden_dim)
        self.convs = torch.nn.ModuleList([
            GATConv(hidden_dim, hidden_dim) for _ in range(self.num_layers - 2)
        ])
        self.conv_out = GATConv(hidden_dim, out_channels)

        # Infomax and correlation loss
        self.infomax = Infomax(1, 1)
        self.loss_corr = 0

    def reset_parameters(self):
        self.conv_in.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv_out.reset_parameters()

    def apply_infomax_loss(self, x, y, i):
        if i % 2 == 0: 
            for _ in range(2): 
                _, f_i, f_j = get_random_dimension_pair(x)
                f_i, f_j = f_i.unsqueeze(1).float(), f_j.unsqueeze(1).float()
                cc3 = correlation_coefficient(f_j, f_i)
                self.loss_corr += (self.infomax.get_loss(f_i, y) + 
                                   self.infomax.get_loss(f_j, y) + 
                                   self.alpha * cc3)

    def forward(self, x, adj_t, y, train_adj=None, test_true=False):
        if not isinstance(y, bool):
            y = y.unsqueeze(1).float() 

        # Input layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_in(x, adj_t)
        x = matrix_norm(x) 
        x = F.relu(x)
        x_init = x

        self.apply_infomax_loss(x, y, i=0)

        # Hidden layers
        for i, conv in enumerate(self.convs, start=1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, adj_t)
            x = matrix_norm(x)
            x = F.relu(x)
            self.apply_infomax_loss(x, y, i)

        # Initial feature flow
        x = x + x_init

        mi = MI_measure(x, y)

        # Output layer
        x = self.conv_out(x, adj_t)

        if test_true:
            corr = torch_corr(x.t())
            corr = torch.triu(corr, 1).abs()
            n = corr.shape[0]
            corr = corr.sum().item() / (n * (n - 1) / 2)
            sim = get_pairwise_sim(x)
            return F.log_softmax(x, dim=1), sim, corr, mi

        return F.log_softmax(x, dim=1)

def get_model(args, dataset):
    data = dataset[0]

    if args.model_type == "BalGNN":
        model = BalGNN(in_dim=data.num_features,
                        hidden_dim=args.hidden_dim,
                        out_channels=dataset.num_classes,
                        dropout=args.dropout,
                        args=args
                        ).cuda()
    elif args.model_type == "BalGNN_GAT":
        model = BalGNN_GAT(in_dim=data.num_features,
                        hidden_dim=args.hidden_dim,
                        out_channels=dataset.num_classes,
                        dropout=args.dropout,
                        args=args
                        ).cuda()
    else:
        raise Exception(f'Wrong model_type')
    
    return model

