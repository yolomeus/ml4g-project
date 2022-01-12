from torch.nn import Module, Linear, ModuleList, ReLU
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, FAConv


class GCN(Module):
    """Simple 2-layer GCN.
    """
    def __init__(self, in_dim, h_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, h_dim)
        self.conv2 = GCNConv(h_dim, out_dim)
        self.dp = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dp, training=self.training)
        x = self.conv2(x, edge_index)

        return x