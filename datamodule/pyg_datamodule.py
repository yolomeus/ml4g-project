from copy import deepcopy

from torch_geometric.data import LightningNodeData


class PyGDataModule(LightningNodeData):
    """A datamodule for wrapping python-geometric Datasets. Returns a single graph """

    def __init__(self, dataset, **kwargs):
        super().__init__(deepcopy(dataset.data), loader='full')
