import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
import os.path as osp
import sys
import tqdm
import itertools
import time

import matplotlib.pyplot as plt

sys.path.insert(0, "..")
sys.path.insert(0, ".")

import auxiliarymethods.multiset_datasets as dp
import preprocessing as pre
import functools

from torch_geometric.datasets import TUDataset
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import shutil
import neural_models.NetGin


class Dataset_data(InMemoryDataset):
    def __init__(
        self,
        root,
        dataset_name,
        multiregression,
        classification,
        use_vertex_labels,
        use_edge_labels,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.dataset_name = dataset_name
        self.classification = classification
        self.multiregression = multiregression
        self.use_vertex_labels = use_vertex_labels
        self.use_edge_labels = use_edge_labels
        super(Dataset_data, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "al_10_10k_2sss"

    @property
    def processed_file_names(self):
        return "al_10_10k_2sss"

    def download(self):
        pass

    @functools.lru_cache
    def process(self):
        data_list = []

        targets = dp.get_dataset(
            self.dataset_name,
            multigregression=self.multiregression,
            classification=self.classification,
        )
        min_targets = min(targets)
        all_indices = [i for i in range(len(targets))]

        node_labels = pre.get_all_tupleset_node_labels_3(
            self.dataset_name, self.use_vertex_labels, self.use_edge_labels
        )
        max_label = max([max(label) for label in node_labels])

        matrices = []
        start_time = time.time()

        matrices = pre.get_local_sparse_tupleset_matrices_3(
            self.dataset_name, all_indices
        )
        end_time = time.time()
        print(f"Time needed for preprocessing: {(end_time - start_time)}")
        for i, m in enumerate(matrices):
            edge_index_1 = torch.tensor(m).t().contiguous()

            data = Data()
            data.edge_index_1 = edge_index_1

            data.x = torch.from_numpy(np.array(node_labels[i])).to(torch.float)
            data.x = data.x.long()

            x_new = torch.zeros(data.x.size(0), max_label + 1)
            x_new[range(x_new.shape[0]), data.x.view(1, data.x.size(0))] = 1
            data.x = x_new
            # if self.dataset_name == "PTC_FM":
            # if targets[i] < 0:
            data.y = (
                torch.from_numpy(np.array([targets[i]])).to(torch.float) - min_targets
            )
            # else:
            #     data.y = torch.from_numpy(np.array([targets[i]])).to(torch.float)
            # data.y = (
            #     torch.from_numpy(np.array([targets[i]])).to(torch.float) - min_targets
            # )
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        return self.num_nodes if key in ["edge_index_1"] else 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def dataset_is_none(dataset):
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)
    return dataset


def test(loader, model, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct, len(loader.dataset)


def train(train_loader, model, optimizer, device):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y.long())
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def preprocess(object):
    def __call__(self, data):
        data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)
        return data


def main(
    epochs: int,
    hidden_units: list,
    dataset_info: str,
    learning_rate: int = 0.001,
    batch_size: int = 32,
    factor: float = 0.5,
    patience: float = 5,
    repetitions: int = 10,
    min_lr: float = 1e-06,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = osp.join(
        osp.dirname(osp.realpath(__file__)),
        ".",
        "data",
        f"{dataset_info[0]}10k",
    )

    dataset = Dataset_data(
        path,
        dataset_name=dataset_info[0],
        multiregression=dataset_info[1],
        classification=dataset_info[2],
        use_vertex_labels=dataset_info[3],
        use_edge_labels=dataset_info[4],
        transform=MyTransform(),
    )
    accuracies = []

    for i in range(repetitions):
        dataset.shuffle()
        test_accuracies = []
        kfold = KFold(n_splits=5, shuffle=True)

        for train_index, test_index in kfold.split(range(len(dataset))):
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_test = 0.0
            all_best_test = 0.0

            train_data = dataset[train_index]
            val_data = dataset[val_index]
            test_data = dataset[test_index]

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

            for hu in hidden_units:
                train_loss_list = []
                val_loss_list = []

                model = neural_models.NetGin.OneGnn(
                    dataset=dataset,
                    hidden_units=hu,
                ).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=factor,
                    patience=patience,
                    min_lr=min_lr,
                )

                loss_arr = []
                for _ in range(1, epochs + 1):
                    learning_rate = scheduler.optimizer.param_groups[0]["lr"]
                    train_loss = train(train_loader, model, optimizer, device)
                    train_loss_list.append(train_loss)

                    cor, len_data = test(val_loader, model, device)
                    val_acc = cor / len_data
                    val_loss_list.append(val_acc)

                    scheduler.step(val_acc)
                    loss_arr.append(val_acc)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        cor, len_data = test(test_loader, model, device)
                        best_test = cor / len_data * 100
                        if best_test > all_best_test:
                            all_best_test = best_test
                    if _ % 1 == 0:
                        print(
                            "Epoch: {:03d}, LR: {:7f}, "
                            "Val Loss: {:.7f}, Test Acc: {:.7f}, Best Test Acc: {:.7f}".format(
                                _, learning_rate, val_acc, best_test, all_best_test
                            )
                        )
                    if learning_rate < min_lr:
                        break
                print(train_loss_list)
                # plt.plot(train_loss_list)
                # # plt.yscale("log")
                # plt.show()
                raise ValueError
            break


if __name__ == "__main__":
    epochs = 100
    repetitions = 1
    hidden_units = [16]
    batch_size = 32
    dataset_name = [
        # ["alchemy_full", True, False, True, True]
        ["ENZYMES", False, True, True, False],
        # ["IMDB-BINARY", False, False, False, False],
        # ["IMDB-MULTI", False, False, False, False],
        # ["PROTEINS", False, True, False, False],
        # ["PTC_FM", False, True, True, False],
    ]
    total_loss = []

    try:
        shutil.rmtree("code/main_methods/datasets")
        shutil.rmtree("code/main_methods/data")
    except:
        pass

    for dataset in dataset_name:
        loss, std = main(
            epochs=epochs,
            hidden_units=hidden_units,
            dataset_info=dataset,
            repetitions=repetitions,
            batch_size=batch_size,
        )
        shutil.rmtree("code/main_methods/datasets")
        shutil.rmtree("code/main_methods/data")
