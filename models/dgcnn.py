
import math
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d, Embedding, BatchNorm1d
from torch_geometric.nn import GCNConv, global_sort_pool, LayerNorm


class DGCNN(torch.nn.Module):
    """An end-to-end deep learning architecture for graph classification, AAAI-18."""

    def __init__(
        self,
        hidden_channels,
        num_layers,
        max_z,
        k=0.6,
        train_dataset=None,
        dynamic_train=False,
        GNN=GCNConv,
        use_feature=False,
        node_embedding=None,
        graph_norm=False,
        batch_norm=False,
        dropout=None,
        seed=42,
    ):
        super(DGCNN, self).__init__()

        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.use_feature = use_feature
        self.node_embedding = node_embedding

        if k <= 1:  # Transform percentile to number.
            if train_dataset is None:
                k = 30
            else:
                if dynamic_train:
                    sampled_train = train_dataset[:1000]
                else:
                    sampled_train = train_dataset
                num_nodes = sorted([g.num_nodes for g in sampled_train])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)
        self.max_z = max_z

        # Fix random seed
        torch.manual_seed(seed)
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.norms = ModuleList()
        self.convs.append(GNN(initial_channels, hidden_channels))
        self.norms.append(LayerNorm(hidden_channels))
        for _ in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
            self.norms.append(LayerNorm(hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))
        self.norms.append(LayerNorm(1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.bn1 = BatchNorm1d(conv1d_channels[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        self.bn2 = BatchNorm1d(conv1d_channels[1])
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        if dense_dim < 0:
            raise ValueError(
                "Negative dimension provided to NN. Increase sortpool_k or decrease hidden_channels and/or num_layers"
            )
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, data, use_feature, embedding, edge_weight=None):
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        x = data.x if use_feature else None
        node_id = data.node_id if embedding else None

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        xs = [x]

        for conv, norm in zip(self.convs, self.norms):
            x_tmp = conv(xs[-1], edge_index, edge_weight)
            if self.graph_norm:
                x_tmp = norm(x_tmp)
            xs += [torch.tanh(x_tmp)]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1d(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)  # p=0.5 in SEAL
        x = self.lin2(x)

        return x

