import torch
from torch_geometric.transforms import BaseTransform, RandomNodeSplit


class BucketizeTransform(BaseTransform):
    def __call__(self, data):
        y = data.y
        # undo log
        y = torch.exp(y)
        y = torch.bucketize(y, torch.tensor([1000, 10000, float('inf')]))
        data.y = y
        return data


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