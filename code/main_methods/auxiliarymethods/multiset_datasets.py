import os.path as osp
import os
from sys import platform

import numpy as np
from torch_geometric.datasets import TUDataset


def read_targets(ds_name):
    if platform == "linux":
        path = "code/main_methods/"
    else:
        path = ""
    # Classes
    with open(
        f"{path}datasets/"
        + ds_name
        + "/"
        + ds_name
        + "/raw/"
        + ds_name
        + "_graph_attributes.txt",
        "r",
    ) as f:
        classes = [float(i) for i in list(f)]
    f.closed

    return np.array(classes)


def read_multi_targets(ds_name):
    if platform == "linux":
        path = "code/main_methods/"
    else:
        path = ""
    # Classes
    with open(
        f"{path}datasets/"
        + ds_name
        + "/"
        + ds_name
        + "/raw/"
        + ds_name
        + "_graph_attributes.txt",
        "r",
    ) as f:
        classes = [[float(j) for j in i.split(",")] for i in list(f)]
    f.closed

    return np.array(classes)


def read_targets(ds_name):
    if platform == "linux":
        path = "code/main_methods/"
    else:
        path = ""
    with open(
        f"{path}datasets/{ds_name}/{ds_name}/raw/{ds_name}_graph_labels.txt"
    ) as f:
        classes = [int(i) for i in list(f)]
    f.closed

    return np.array(classes)


def get_dataset(dataset, multigregression=False, classification=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "datasets", dataset)
    TUDataset(path, name=dataset)
    if classification:
        return read_targets(dataset)
    if multigregression:
        return read_multi_targets(dataset)
    else:
        return read_targets(dataset)
