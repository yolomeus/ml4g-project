import os
import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torch_geometric.data import Data
from torch_geometric.datasets import Actor, Planetoid

from datamodule.transform import RandomSplitTransform


class PlanetoidDataset(Planetoid):
    def __init__(self, num_features: int, num_classes: int, root: str, name: str, **kwargs):
        """Wrapper for Planetoid Dataset that adds num_features and num_classes for automatically infering dimensions
        based on the dataset during runtime.

        Args:
            num_features: number of features for each node in the dataset.
            num_classes: number of target classes for each node.
        """

        super().__init__(root, name, **kwargs)
        self._num_features = num_features
        self._num_classes = num_classes

    def download(self):
        super().download()

    def process(self):
        super().process()


class WikiActorDataset(Actor):
    def __init__(self, num_features: int, num_classes: int, root: str, name: str):
        """

        Args:
            num_features:
            num_classes:
            root:
            name:
        """

        super().__init__(root, pre_transform=RandomSplitTransform(num_classes))

        self._num_features = num_features
        self._num_classes = num_classes
        self._name = name

    def download(self):
        super().download()

    def process(self):
        super().process()


class WikiRawDataset(Dataset):

    def __init__(self, num_features: int, num_classes: int, root: str, name: str):
        self._num_features = num_features
        self._num_classes = num_classes
        self._name = name

        self.root = root
        self.out_dir = os.path.join(root, 'processed/')
        self.processed_file = os.path.join(self.out_dir, 'processed.pkl')

        if not os.path.exists(self.processed_file):
            self.preprocess()

        with open(self.processed_file, 'rb') as fp:
            self.data = pickle.load(fp)

    def __getitem__(self, index) -> T_co:
        return self.data

    def __len__(self):
        return 1

    def preprocess(self):
        os.makedirs(self.out_dir, exist_ok=True)
        edge_file = os.path.join(self.root, 'raw/edges.txt')
        feature_file = os.path.join(self.root, 'raw/features.txt')
        label_file = os.path.join(self.root, 'raw/labels.txt')

        edges = torch.from_numpy(pd.read_csv(edge_file, delimiter=' ', header=None).to_numpy().T).long()
        features = torch.from_numpy(pd.read_csv(feature_file, delimiter=' ', header=None).to_numpy()).float()
        labels = torch.from_numpy(pd.read_csv(label_file, header=None).to_numpy()).squeeze().long()

        self.data = Data(features, edges, y=labels)
        self.data = RandomSplitTransform(self._num_classes)(self.data)

        with open(self.processed_file, 'wb') as fp:
            pickle.dump(self.data, fp)
