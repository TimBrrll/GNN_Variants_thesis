import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
import shutil

import time
from neural_models.GcnSum import GcnSum
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


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

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()


def preprocess(object):
    def __call__(self, data):
        data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)
        return data


def main(
    epochs: int,
    layers: list,
    hidden_units: list,
    dataset_name: str,
    batch_size: int = 32,
    n_folds: int = 5,
    factor: float = 0.5,
    patience: float = 5,
    repetitions: int = 10,
    min_lr: float = 1e-06,
):
    dataset = TUDataset(root=f"../tmp/{dataset_name}", name=dataset_name).shuffle()
    dataset = dataset_is_none(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accuracies = []

    for i in range(repetitions):
        dataset.shuffle()

        test_accuracies = []
        kfold = KFold(n_splits=n_folds, shuffle=True)

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

            for layer in layers:
                for hu in hidden_units:
                    model = GcnSum(dataset=dataset, layers=layer, hidden_units=hu).to(
                        device
                    )

                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=factor,
                        patience=patience,
                        min_lr=min_lr,
                    )

                    for _ in range(1, epochs + 1):
                        learning_rate = scheduler.optimizer.param_groups[0]["lr"]
                        train(train_loader, model, optimizer, device)
                        cor, len_data = test(val_loader, model, device)
                        val_acc = cor / len_data
                        scheduler.step(val_acc)

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            cor, len_data = test(test_loader, model, device)
                            best_test = cor / len_data * 100
                            if best_test > all_best_test:
                                all_best_test = best_test
                        if _ % 50 == 0:
                            print(
                                "Epoch: {:03d}, LR: {:7f}, "
                                "Val Loss: {:.7f}, Test Acc: {:.7f}, Best Test Acc: {:.7f}".format(
                                    _, learning_rate, val_acc, best_test, all_best_test
                                )
                            )
                        if learning_rate < min_lr:
                            break

            test_accuracies.append(best_test)
        accuracies.append(float(np.array(test_accuracies).mean()))
    return np.array(accuracies).mean(), np.array(accuracies).std()


if __name__ == "__main__":
    epochs = 100
    layers = [3, 4, 5]
    repetitions = 5
    hidden_units = [32, 64, 128]
    dataset_name = ["ENZYMES", "PROTEINS", "IMDB-BINARY", "IMDB-MULTI", "PTC_FM"]
    batch_size = 32
    preprocess_bool = False
    n_folds = 5

    for dataset in dataset_name:
        print(f"------------------------- Dataset: {dataset} ----------------------")

        loss, std = main(
            epochs=epochs,
            layers=layers,
            hidden_units=hidden_units,
            dataset_name=dataset,
            repetitions=repetitions,
            batch_size=batch_size,
            n_folds=n_folds,
        )
        print("#####################################################")

        print(
            f"FINAL RESULT GCNSUM for {dataset}: mean_losses: {loss}, std_losses: {std}"
        )
        shutil.rmtree("../tmp")

    big_dataset_names = [
        "Yeast",
        "YeastH",
        "UACC257",
        "UACC257H",
        "OVCAR-8",
        "OVCAR-8H",
    ]

    big_data_reps = 3
    big_data_layers = [3]
    big_data_hu = [64]
    big_data_epochs = 100
    batch_size = 64

    for dataset in big_dataset_names:
        print(f"------------------------- Dataset: {dataset} ----------------------")
        loss, std = main(
            epochs=big_data_epochs,
            layers=big_data_layers,
            hidden_units=big_data_hu,
            dataset_name=dataset,
            repetitions=big_data_reps,
            batch_size=batch_size,
        )
        print("#####################################################")
        print(
            f"FINAL RESULT GCNSUM for {dataset}: mean_losses: {loss}, std_losses: {std}"
        )
        print("#####################################################")
        shutil.rmtree("../tmp")
