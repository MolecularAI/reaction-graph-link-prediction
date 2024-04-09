
import random
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path

import torch
from torch_geometric.data import Data, Dataset


class SEALDynamicDataset(Dataset):
    """Class for creating dataset used for link prediction with SEAL.
    This class constructs subgraphs for each target link.
    """

    def __init__(self, root, dataset, settings, split="train", **kwargs):
        self.data = dataset.data
        self.num_nodes = dataset.num_nodes
        self.num_hops = settings["num_hops"]
        self.node_label = settings["node_label"]
        self.ratio_per_hop = settings["ratio_per_hop"]
        self.max_nodes_per_hop = settings["max_nodes_per_hop"]
        self.split = split
        super(SEALDynamicDataset, self).__init__(root)

        # Get positive and negative edges for given split
        self.pos_edge = self.data.split_edge[self.split]["pos"]
        self.neg_edge = self.data.split_edge[self.split]["neg"]

        # Create a torch with positive and negative edges and one with labels
        self.links = torch.cat([self.pos_edge, self.neg_edge], 1).t().tolist()
        self.labels = [1] * self.pos_edge.size(1) + [0] * self.neg_edge.size(1)

        edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)

        self.A = ssp.csr_matrix(
            (
                edge_weight.numpy(),
                (self.data.edge_index[0].numpy(), self.data.edge_index[1].numpy()),
            ),
            shape=(self.num_nodes, self.num_nodes),
        )

    def __len__(self):
        return len(self.links)

    def get(self, idx):
        """Retrieves a subgraph around the source and target nodes, given by idx."""

        links, labels = self.links, self.labels

        tmp = k_hop_subgraph(
            links[idx],
            self.num_hops,
            self.A,
            self.ratio_per_hop,
            self.max_nodes_per_hop,
            node_features=self.data.x,
            y=labels[idx],
        )

        data = construct_pyg_graph(*tmp, links[idx], self.node_label)

        return data


### UTILS ###
def k_hop_subgraph(
    node_idx,
    num_hops,
    A,
    sample_ratio=1.0,
    max_nodes_per_hop=None,
    node_features=None,
    y=1,
):
    """Extract the k-hop enclosing subgraph around link node_idx=(src, dst) from A."""

    dists = [0, 0]
    visited = set(node_idx)
    fringe = set(node_idx)
    for dist in range(1, num_hops + 1):
        # Get 1-hop neighbors not visited
        fringe = set(A[list(fringe)].indices)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio * len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        node_idx = node_idx + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[node_idx, :][:, node_idx]

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[node_idx]

    return node_idx, subgraph, node_features, y


def construct_pyg_graph(node_ids, adj, node_features, y, link, node_label="drnl"):
    """Construct a pytorch_geometric graph from a scipy csr adjacency matrix."""

    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]

    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])

    if node_label == "drnl":  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
    else:
        raise NotImplementedError(f"{node_label} is not a valid 'node_label' setting.")

    sub_data = Data(
        node_features,
        edge_index,
        edge_weight=edge_weight,
        y=y,
        z=z,
        node_id=node_ids,
        num_nodes=num_nodes,
        link=link,
    )
    return sub_data


def drnl_node_labeling(adj, src, dst):
    """Double Radius Node Labeling (DRNL)."""
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(
        adj_wo_src, directed=False, unweighted=True, indices=dst - 1
    )
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.0
    z[dst] = 1.0
    z[torch.isnan(z)] = 0.0

    return z.to(torch.long)

