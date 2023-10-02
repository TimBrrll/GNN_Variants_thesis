import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
import os.path as osp
import sys
import tqdm
import itertools
import time

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

        node_labels = pre.get_all_tupleset_node_labels_2_2(
            self.dataset_name, self.use_vertex_labels, self.use_edge_labels
        )
        max_label = max([max(label) for label in node_labels])
        matrices = []
        start_time = time.time()

        matrices = pre.get_all_tupleset_matrices_2(self.dataset_name, all_indices)
        end_time = time.time()
        print(
            f"Time needed for preprocessing for {dataset_name}: {(end_time - start_time)}"
        )

        # torch.save((data, slices), self.processed_paths[0])


def main(dataset_name, multiregression, classification, repetitions):
    targets = dp.get_dataset(
        dataset_name,
        multigregression=multiregression,
        classification=classification,
    )
    times = []
    all_indices = [i for i in range(len(targets))]
    for i in range(repetitions):
        start_time = time.time()
        matrices = pre.get_local_sparse_multiset_matrices_3(dataset_name, all_indices)
        end_time = time.time()
        times.append(end_time - start_time)

    print(
        f"For M-3-GNN preprocessing for dataset {dataset_name} needed: {float(np.array(times).mean())} seconds"
    )


if __name__ == "__main__":
    epochs = 100
    repetitions = 10
    hidden_units = [32, 64, 128]
    batch_size = 32
    dataset_name = [
        # ["ENZYMES", False, True, True, False],
        ["IMDB-BINARY", False, False, False, False],
        # ["IMDB-MULTI", False, False, False, False],
        # ["PROTEINS", False, True, True, False],
        # # ["REDDIT-BINARY", False, False, False, False],
        # ["PTC_FM", False, True, True, False],
    ]

    try:
        shutil.rmtree("datasets")
        shutil.rmtree("data")
    except:
        pass

    for dataset in dataset_name:
        print(f"------------------------- M-3-GNN ----------------------")
        main(
            dataset_name=dataset[0],
            multiregression=dataset[1],
            classification=dataset[2],
            repetitions=repetitions,
        )
        # shutil.rmtree("datasets")
        # shutil.rmtree("data")
