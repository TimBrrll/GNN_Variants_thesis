import time
import os.path as osp
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, Set2Set
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import torch.nn.functional as F
from torch_geometric.nn.conv.graph_conv import GraphConv
from torch_scatter import scatter_mean


class OneGnn(torch.nn.Module):
    def __init__(self, dataset, hidden_units):
        super(OneGnn, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, hidden_units)
        self.conv2 = GraphConv(hidden_units, hidden_units * 2)
        self.conv3 = GraphConv(hidden_units * 2, hidden_units * 2)
        self.mlp1 = torch.nn.Linear(hidden_units * 2, hidden_units * 2)
        self.mlp2 = torch.nn.Linear(hidden_units * 2, hidden_units)
        self.mlp3 = torch.nn.Linear(hidden_units, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
        self.mlp3.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index_1, data.batch
        edge_index = edge_index.long()

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))

        x = scatter_mean(x, batch, dim=0)

        x = F.elu(self.mlp1(x))
        x = F.dropout(x, p=0.5)
        x = F.elu(self.mlp2(x))
        x = self.mlp3(x)

        return F.log_softmax(x, dim=1)
