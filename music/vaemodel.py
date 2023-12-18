
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch_geometric
import pickle
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
import numpy as np
from torch_geometric.utils import to_dense_batch
from torch import optim
import torch_geometric.nn as geom_nn
import random 
import copy
from predictrel import RelPredictor

# Define the size of the node features and the number of edge features
magenta_size = 256
num_features = magenta_size
num_edge_features = 49

# Message passing neural network layer using continuous kernel-based convolution
class NNConv(MessagePassing):
    def __init__(self, num_features, num_edge_features, in_channels, out_channels, aggr='add', root_weight=True, bias=True, **kwargs):
        super(NNConv, self).__init__(aggr=aggr, **kwargs)
        self.n_features = num_features
        self.in_channels = in_channels
        self.out_channels = 20
        self.nn = nn.Sequential(nn.Linear(num_edge_features, self.n_features * self.out_channels))
        self.aggrnn = nn.Sequential(nn.Linear(num_features + self.out_channels, num_features), nn.ReLU(), nn.Linear(num_features, num_features))

        if root_weight:
            self.root = Parameter(torch.randn(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels)) 
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index, edge_attr, prin=False):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        a = self.propagate(edge_index, x=x, pseudo=pseudo)
        if prin:
            print(a[0, :10])
        return a

    def message(self, x_j, pseudo):
        weight = self.nn(pseudo).view(-1, self.n_features, self.out_channels)
        return nn.Sigmoid()(torch.matmul(x_j.unsqueeze(1), weight).squeeze(1))

    def update(self, aggr_out, x):
        y = torch.cat((aggr_out, x), axis=1)
        return self.aggrnn(y)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

# Function to compute the covariance matrix
def cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov

# Encoder module of a Variational Autoencoder (VAE)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = NNConv(num_features, num_edge_features, num_edge_features, num_features)
        self.conv2 = NNConv(num_features, num_edge_features, num_edge_features, num_features)
        self.mu = nn.Linear(num_features, 200)
        self.logvar = nn.Linear(num_features, 200)

    def forward(self, x, edge_ind, edge_attr):
        x = self.conv1(x, edge_ind, edge_attr)
        x = self.conv2(x, edge_ind, edge_attr)
        x_mu = self.mu(x)
        x_log = self.logvar(x)
        return x_mu, x_log

# Decoder module of a Variational Autoencoder (VAE)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = NNConv(num_features, num_edge_features, num_edge_features, num_features)
        self.conv2 = NNConv(num_features, num_edge_features, num_edge_features, num_features)
        self.autodec = nn.Sequential(nn.Linear(200, num_features), nn.ReLU(), nn.Linear(num_features, num_features))

    def forward(self, x, edge_ind, edge_attr):
        x = self.autodec(x)
        x = self.conv1(x, edge_ind, edge_attr)
        x = self.conv2(x, edge_ind, edge_attr)
        return x

# Variational Autoencoder (VAE) model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.rel_predict = RelPredictor()
        self.rel_predict.load_state_dict(torch.load("graphnn/relpredictor.pth"))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index, edge_attr):
        x_mu, x_log = self.encoder(x, edge_index, edge_attr)
        z = self.reparameterize(x_mu, x_log)
        x = self.decoder(z, edge_index, edge_attr)
        edge_x = torch.cat([x[edge_index[0, :]][:, :magenta_size], x[edge_index[1, :]][:, :magenta_size]], axis=1)
        attr_predict = self.rel_predict(edge_x)

        return x, z, x_mu, x_log, attr_predict
