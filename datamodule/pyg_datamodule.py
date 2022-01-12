from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.datasets import Planetoid

from datamodule.default_datamodule import AbstractDefaultDataModule


class PyGDataModule(AbstractDefaultDataModule):
    """A datamodule for wrapping python-geometric Datasets. Returns a single graph """

    def __init__(self, base_dir, dataset: Dataset, train_conf, test_conf, num_workers, pin_memory):
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
        super().__init__(root, name)
        self._num_features = num_features
        self._num_classes = num_classes
