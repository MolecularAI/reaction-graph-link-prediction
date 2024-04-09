
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, seed, dropout):
        super(GCNEncoder, self).__init__()
        self.dropout = dropout
        # Fix random seed
        torch.manual_seed(seed)
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, seed, dropout):
        super(VariationalGCNEncoder, self).__init__()
        self.dropout = dropout
        # Fix random seed
        torch.manual_seed(seed)
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, seed, dropout):
        super(LinearEncoder, self).__init__()
        self.dropout = dropout
        # Fix random seed
        torch.manual_seed(seed)
        self.conv = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, seed, dropout):
        super(VariationalLinearEncoder, self).__init__()
        self.dropout = dropout
        # Fix random seed
        torch.manual_seed(seed)
        self.conv_mu = GCNConv(in_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

