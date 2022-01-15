import os.path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Actor
from torch_geometric.transforms import BaseTransform, RandomNodeSplit

from datamodule.default_datamodule import AbstractDefaultDataModule


class PyGDataModule(AbstractDefaultDataModule):
    """A datamodule for wrapping python-geometric Datasets. Returns a single graph """

    def __init__(self, base_dir, dataset, train_conf, test_conf, num_workers, pin_memory):
        super().__init__(train_conf, test_conf, num_workers, pin_memory)

        self.base_dir = base_dir
        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self._train_conf.batch_size,
                          num_workers=self._num_workers,
                          pin_memory=self._pin_memory,
                          persistent_workers=self._num_workers > 0,
                          collate_fn=self._collate)

    def test_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self._test_conf.batch_size,
                          num_workers=self._num_workers,
                          pin_memory=self._pin_memory,
                          persistent_workers=self._num_workers > 0,
                          collate_fn=self._collate)

    def val_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self._test_conf.batch_size,
                          num_workers=self._num_workers,
                          pin_memory=self._pin_memory,
                          persistent_workers=self._num_workers > 0,
                          collate_fn=self._collate)

    @staticmethod
    def _collate(x):
        return x[0]


class PlanetoidDataset(Planetoid):
    def __init__(self, num_features: int, num_classes: int, root: str, name: str):
        """Wrapper for Planetoid Dataset that adds num_features and num_classes for automatically infering dimensions
        based on the dataset during runtime.

        Args:
            num_features: number of features for each node in the dataset.
            num_classes: number of target classes for each node.
        """
        Planetoid(root, name)  # don't ask
        super().__init__(root, name)
        self._num_features = num_features
        self._num_classes = num_classes


class WikiRawDataset(Dataset):

    def __init__(self, num_features: int, num_classes: int, root: str, name: str):
        self._num_features = num_features
        self._num_classes = num_classes
        self._name = name

        edge_file = os.path.join(root, 'edges.txt')
        feature_file = os.path.join(root, 'features.txt')
        label_file = os.path.join(root, 'labels.txt')

        edges = torch.tensor(pd.read_csv(edge_file, delimiter=' ', header=None).to_numpy().T, dtype=torch.long)
        features = torch.tensor(pd.read_csv(feature_file, delimiter=' ', header=None).to_numpy(),
                                dtype=torch.float32)
        labels = torch.tensor(pd.read_csv(label_file, header=None).to_numpy(), dtype=torch.long).squeeze()

        self.data = Data(features, edges, y=labels)
        self.data = RandomSplitTransform(self._num_classes)(self.data)

    def __getitem__(self, index) -> T_co:
        return self.data

    def __len__(self):
        return 1


class WikiActorDataset(Actor):
    def __init__(self, num_features: int, num_classes: int, root: str, name: str):
        """

        Args:
            num_features:
            num_classes:
            root:
            name:
        """

        Actor(root, pre_transform=RandomSplitTransform(num_classes))
        super().__init__(root, pre_transform=RandomSplitTransform(num_classes))

        self._num_features = num_features
        self._num_classes = num_classes
        self._name = name


class RandomSplitTransform(BaseTransform):
    def __init__(self, num_classes, train_percent=.6):
        self.train_percent = train_percent
        self.num_classes = num_classes

    def __call__(self, data):
        n_total = len(data.x)
        n_test = int(.2 * n_total)
        n_train_per_c = int((n_total * self.train_percent) / self.num_classes)
        data = RandomNodeSplit(num_train_per_class=n_train_per_c, num_val=n_test, num_test=n_test)(data)
        return data


class BucketizeTransform(BaseTransform):
    def __call__(self, data):
        y = data.y
        # undo log
        y = torch.exp(y)
        y = torch.bucketize(y, torch.tensor([1000, 10000, float('inf')]))
        data.y = y
        return data
