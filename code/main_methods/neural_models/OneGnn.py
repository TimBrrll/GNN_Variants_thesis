import torch
import torch.nn.functional as F

from torch_geometric.nn import GraphConv
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
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = scatter_mean(x, batch, dim=0)

        x = F.relu(self.mlp1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)
        return F.log_softmax(x, dim=-1)
