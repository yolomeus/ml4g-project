from torch.utils.data import DataLoader
from torch_geometric.data import Dataset

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
                          collate_fn=lambda x: x[0])

    def test_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self._test_conf.batch_size,
                          num_workers=self._num_workers,
                          pin_memory=self._pin_memory,
                          persistent_workers=self._num_workers > 0,
                          collate_fn=lambda x: x[0])

    def val_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self._test_conf.batch_size,
                          num_workers=self._num_workers,
                          pin_memory=self._pin_memory,
                          persistent_workers=self._num_workers > 0,
                          collate_fn=lambda x: x[0])
