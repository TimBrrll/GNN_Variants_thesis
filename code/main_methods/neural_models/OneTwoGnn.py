import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

from torch_scatter import scatter_mean
from torch_geometric.nn import NNConv, GraphConv
from k_gnn import TwoMalkin, avg_pool, GraphConv


class MyPretransform(object):

    def __call__(self, data):
        x = data.x
        data.x = data.x[:, :5]
        data = TwoMalkin()(data)
        data.x = x
        return data


class OneTwoGnn(torch.nn.Module):

    def __init__(self, dataset, hidden_units, iso_type_2):
        super(OneTwoGnn, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, 32)
        self.conv2 = GraphConv(hidden_units, hidden_units * 2)
        self.conv3 = GraphConv(hidden_units * 2, hidden_units * 2)
        self.conv4 = GraphConv(hidden_units * 2 + iso_type_2, hidden_units * 2)
        self.conv5 = GraphConv(hidden_units * 2, hidden_units * 2)
        self.mlp1 = torch.nn.Linear(2 * hidden_units * 2, hidden_units * 2)
        self.mlp2 = torch.nn.Linear(hidden_units * 2, hidden_units)
        self.mlp3 = torch.nn.Linear(hidden_units, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
        self.mlp3.reset_parameters()

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index))
        data.x = F.elu(self.conv2(data.x, data.edge_index))
        data.x = F.elu(self.conv3(data.x, data.edge_index))
        x = data.x
        x_1 = scatter_mean(data.x, data.batch, dim=0)

        data.x = avg_pool(x, data.assignment_index_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)

        data.x = F.elu(self.conv4(data.x, data.edge_index_2))
        data.x = F.elu(self.conv5(data.x, data.edge_index_2))
        x_2 = scatter_mean(data.x, data.batch_2, dim=0)

        x = torch.cat([x_1, x_2], dim=1)

        x = F.elu(self.mlp1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.mlp2(x))
        x = self.mlp3(x)
        return F.log_softmax(x, dim=1)