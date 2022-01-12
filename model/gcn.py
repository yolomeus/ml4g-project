from torch.nn import Module, Linear, ModuleList, ReLU, Dropout
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


class FAGCN(Module):
    def __init__(self, in_dim, h_dim, out_dim, eps, dropout, n_layers, activation: Module = ReLU()):
        super().__init__()
        self.lin_in = Linear(in_dim, h_dim)
        self.layers = ModuleList([FAConv(h_dim, eps)
                                  for _ in range(n_layers)])
        self.lin_out = Linear(h_dim, out_dim)

        self.act = activation
        self.dp = Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.lin_in(x)
        x_0 = self.act(x)
        x = x_0
        for layer in self.layers:
            x = self.dp(x)
            x = layer(x, x_0, edge_index)

        return self.lin_out(x)
