from torch.nn import Module, Linear, ModuleList, ReLU, Dropout
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
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
    def __init__(self, in_dim, h_dim, out_dim, eps, dropout, n_layers):
        super().__init__()
        self.lin_in = Linear(in_dim, h_dim)
        self.layers = ModuleList([FAConv(h_dim, eps, dropout)
                                  for _ in range(n_layers)])
        self.lin_out = Linear(h_dim, out_dim)

        self.act = ReLU()
        self.dp = Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        xavier_normal_(self.lin_in.weight, gain=1.414)
        xavier_normal_(self.lin_out.weight, gain=1.414)
        for layer in self.layers:
            xavier_normal_(layer.att_l.weight, gain=1.414)
            xavier_normal_(layer.att_r.weight, gain=1.414)

    def forward(self, x, edge_index):
        x = self.dp(x)
        x = self.lin_in(x)
        x = self.act(x)

        x = self.dp(x)
        x_0 = x
        for layer in self.layers:
            x = layer(x, x_0, edge_index)

        return self.lin_out(x)
