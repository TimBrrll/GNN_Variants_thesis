import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU
from torch.nn import BatchNorm1d as BN


class GIN(torch.nn.Module):
    def __init__(self, dataset, layers, hidden_units, train_eps: bool = False):
        super(GIN, self).__init__()
        self.conv = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden_units),
                ReLU(),
                Linear(hidden_units, hidden_units),
                ReLU(),
                BN(hidden_units),
            ),
            train_eps=train_eps,
        )
        self.convs = torch.nn.ModuleList()
        for i in range(layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_units, hidden_units),
                        ReLU(),
                        Linear(hidden_units, hidden_units),
                        ReLU(),
                        BN(hidden_units),
                    ),
                    train_eps=train_eps,
                )
            )
        self.first_lin = Linear(hidden_units, hidden_units)
        self.second_lin = Linear(hidden_units, dataset.num_classes)

    def reset_parameters(self):
        self.conv.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.first_lin.reset_parameters()
        self.second_lin.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.second_lin(x)
        return F.log_softmax(x, dim=-1)
