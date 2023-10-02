import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, MessagePassing, global_mean_pool
from torch.nn import Sequential, ModuleList, Linear, ReLU


class GraphSAGE(MessagePassing):

    def __init__(self, dataset, layers, hidden_units):
        super().__init__()
        self.conv = SAGEConv(dataset.num_features, hidden_units)
        self.convs_layers = ModuleList()

        for _ in range(layers - 1):
            self.convs_layers.append(SAGEConv(hidden_units, hidden_units))

        self.first_lin = Linear(hidden_units, hidden_units)
        self.second_lin = Linear(hidden_units, dataset.num_classes)

    def reset_parameters(self):
        self.conv.reset_parameters()
        for layer in self.convs_layers:
            layer.reset_parameters()
        self.first_lin.reset_parameters()
        self.second_lin.reset_parameters()

    def forward(self, input):
        x, edge_index, batch = input.x, input.edge_index, input.batch
        x = F.relu(self.conv(x, edge_index))
        for layer in self.convs_layers:
            x = F.relu(layer(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.second_lin(x)
        return F.log_softmax(x, dim=-1)
