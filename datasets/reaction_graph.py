import logging
import sys
import warnings
import os

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected

from utils.negative_sampling import (
    sample_random,
    correct_overlaps,
    sample_distribution,
    sample_degree_preserving_distribution,
    sample_analogs,
)

# Import Graph-tool, ignore warnings related to C++ code conversion
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import graph_tool.all as gt


class ReactionGraph(InMemoryDataset):
    def __init__(self, root, settings, seed_neg_sampling, **kwargs):
        self.settings = settings

        self.percent_edges = settings["train_fraction"]
        if "test_fraction" in settings.keys():
            self.percent_edges_test = settings["test_fraction"]
        else:
            self.percent_edges_test = settings["train_fraction"]

        self.neg_pos_ratio = settings["neg_pos_ratio"]
        self.neg_pos_ratio_test = settings["neg_pos_ratio_test"]
        self.splitting = self.settings["splitting"]

        self.seed = seed_neg_sampling
        self.fold = None
        self.num_nodes = None

        super(ReactionGraph, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.settings["graph_path"]]

    @property
    def processed_file_names(self):
        return [f"processed_data.pt"]

    def download(self):
        pass

    def process(self):
        # Import graph with graph-tool
        graph = gt.load_graph(self.settings["graph_path"])
        self.num_nodes = graph.num_vertices()

        attribute = "fingerprint" if self.settings["use_attribute"] else None
        # Node matrix with optional attributes
        if attribute == "fingerprint":
            # Node attributes are fingerprints of size 1024
            attrs = graph.vertex_properties[attribute].get_2d_array(
                list(range(0, 1024))
            )
            X = torch.tensor(attrs.transpose())
        else:
            X = None

        # Create edge adjacency list
        adj_list = graph.get_edges(
            eprops=[
                graph.edge_properties["random fold"]
            ]
        )

        A = torch.LongTensor(adj_list[:, :2].transpose())

        # Create data object with edges and node attributes
        self.data = Data(x=X, edge_index=A)

        # Add folds to use when creating train/val/test splits
        self.data.random_folds = adj_list[:, 2]

        # Confirm upper triangular graph
        row, col = self.data.edge_index
        l = len(row)
        self.data.edge_index = None
        mask = row < col
        row, col = row[mask], col[mask]
        if l != len(row):
            logging.warning("The graph is not upper triangular.")
            sys.exit()

        # All positive edges in full graph
        self.data.pos_edge = torch.stack([row, col], dim=0).long()

        # Find how many negative edges to sample by 'sample_distribution' and 'sample_random'
        # respectively.
        if self.splitting == "random":
            self.fold = self.data.random_folds

        fraction_test = np.sum(self.fold == 10) / len(self.fold)

        # n_neg_to_sample = int(((fraction_test * self.neg_pos_ratio_test * self.percent_edges_test) \
        # + ((1 - fraction_test) * self.neg_pos_ratio)) \
        # + ((1 - fraction_test) * self.neg_pos_ratio * self.percent_edges)) \
        # * self.data.pos_edge.shape[1])
        n_neg_to_sample = int(
            (
                (fraction_test * self.neg_pos_ratio_test)
                + ((1 - fraction_test) * self.neg_pos_ratio)
            )
            * self.data.pos_edge.shape[1]
        )
        n_dist_negs_to_sample = int(
            n_neg_to_sample * self.settings["fraction_dist_neg"]
        )
        n_rand_negs_to_sample = int(
            n_neg_to_sample * (1 - self.settings["fraction_dist_neg"])
        )

        # Sample from node distribution of the positive edges
        if n_dist_negs_to_sample > 0:
            i_dist, j_dist = sample_degree_preserving_distribution(
                "data/negative_degree-preserving_distribution_graph="
                + os.path.splitext(os.path.basename(self.settings["graph_path"]))[0]
                + "_seed="
                + str(self.seed)
                + ".pt",
                self.data.pos_edge,
                n_dist_negs_to_sample,
                self.num_nodes,
                self.data.pos_edge,
                seed=self.seed,
            )
        else:
            i_dist, j_dist = torch.tensor([]), torch.tensor([])

        self.data.neg_edge_dist = torch.stack((i_dist, j_dist), dim=0)

        # Sample randomly from all non-isolated nodes
        if n_rand_negs_to_sample > 0:
            i_rand, j_rand = sample_random(
                self.data.pos_edge,
                n_rand_negs_to_sample,
                self.num_nodes,
                self.data.pos_edge,
                seed=self.seed,
            )

            # Correct overlap between 2 sets of negatives
            i_rand, j_rand = correct_overlaps(
                (i_dist, j_dist), (i_rand, j_rand), self.num_nodes, seed=self.seed
            )
        else:
            i_rand, j_rand = torch.tensor([]), torch.tensor([])
        self.data.neg_edge_rand = torch.stack((i_rand, j_rand), dim=0)

        # Store data objects
        torch.save(self.collate([self.data]), self.processed_paths[0])

    def create_negative_set(self):
        """Creates the negative set from the randomly sampled and sampled from
        distribution negatives, according to the given percentage."""

        n_neg_edges = (
            self.data.neg_edge_rand.shape[1] + self.data.neg_edge_dist.shape[1]
        )
        n_dist = self.data.neg_edge_dist.shape[1]

        np.random.seed(seed=self.seed)
        mask = n_dist > np.random.permutation(n_neg_edges)

        # Merge and mix negative edges and negative edges sampled from distribution to one tensor
        # stored as: 'self.data.neg_edge'
        neg_edge = np.zeros((n_neg_edges, 2))
        neg_edge[mask] = self.data.neg_edge_dist.transpose(0, 1).detach().clone()
        neg_edge[~mask] = self.data.neg_edge_rand.transpose(0, 1).detach().clone()

        self.data.neg_edge = (
            torch.tensor(neg_edge).transpose(1, 0).int().detach().clone()
        )

    def process_splits(self):
        """Create the train, validation and test sets."""
        valid_fold = self.settings["valid_fold"]

        if valid_fold not in self.fold:
            sys.exit(
                f'Invalid setting "valid_fold" = {valid_fold}. Choose from: {set(self.fold) - set([10])}'
            )

        num_edges = self.data.pos_edge.shape[1]

        # Prepare edge split based on time or random folds
        # According to split labels:
        # 0 = 'train', 1 = 'valid', 2 = 'test', 20 = not included in any split (time)

        if self.splitting == "random":
            self.fold = self.data.random_folds[0]
            # Initialize all edges as to belong in train set
            edge_split_label = np.zeros(num_edges, dtype=int)
            # Test set
            edge_split_label[self.fold == 10] = 2
            # Validation set
            edge_split_label[self.fold == valid_fold] = 1

        # Create self.data.neg_edge
        self.create_negative_set()

        n_pos_all = int(self.data.pos_edge.shape[1])
        n_pos = int(self.data.pos_edge.shape[1] * self.percent_edges)

        # Split edges between train / valid / test sets
        self.data.split_edge = {"train": {}, "valid": {}, "test": {}}
        lower_range = 0
        for key, split in {2: "test", 1: "valid", 0: "train"}.items():
            logging.debug("Creating %s split.", split.upper())

            # Create the positive set
            all_pos_edges_in_split = (
                self.data.pos_edge[:, edge_split_label == key].detach().clone()
            )
            # Extracting a subset of all positive edges in split
            n_in_split = all_pos_edges_in_split.shape[1]
            perm_split_percent = np.random.permutation(n_in_split)
            if split != "test":
                perm_pos = perm_split_percent[: int(n_in_split * self.percent_edges)]
            else:
                perm_pos = perm_split_percent[
                    : int(n_in_split * self.percent_edges_test)
                ]
            self.data.split_edge[split]["pos"] = (
                all_pos_edges_in_split[:, perm_pos].detach().clone()
            )

            # Create the negative set
            if split == "train":
                train_neg_set = []
                for fold in range(n_folds):
                    if fold != valid_fold:
                        train_neg_set.append(
                            self.data.neg_edge[
                                :, folds2index[fold][0] : folds2index[fold][1]
                            ]
                        )
                self.data.split_edge[split]["neg"] = torch.cat(train_neg_set, dim=1)
            elif split == "valid":
                self.data.split_edge[split]["neg"] = self.data.neg_edge[
                    :, folds2index[valid_fold][0] : folds2index[valid_fold][1]
                ]

            elif split == "test":
                # Create the negative set index dictionary
                # n_neg_in_test = int(self.data.split_edge['test']['pos'].shape[1] * self.neg_pos_ratio_test)
                folds2index = {
                    "test": (0, all_pos_edges_in_split.shape[1])
                }  # n_neg_in_test)}
                first_index = all_pos_edges_in_split.shape[1]  # n_neg_in_test
                # Test fold has fixed ID 10, so take next largest ID (4 or 9) + 1 for 5 or 10 fold cross validation
                n_folds = sorted(set(self.fold.tolist()))[-2] + 1
                for _fold in range(n_folds):
                    n_in_fold = len(edge_split_label[self.fold == _fold])
                    folds2index[_fold] = (
                        first_index,
                        first_index
                        + int(n_in_fold * self.neg_pos_ratio * self.percent_edges),
                    )
                    first_index += int(n_in_fold * self.neg_pos_ratio)
                if (
                    folds2index[n_folds - 1][1] > self.data.neg_edge.shape[1] + 1
                ):  # second should be +1 first, not sure why the +1
                    logging.warning(
                        "Last index of last fold for negative sampling > total number of negatives sampled."
                    )

                self.data.split_edge[split]["neg"] = self.data.neg_edge[
                    :,
                    folds2index["test"][0] : int(
                        folds2index["test"][1] * self.percent_edges_test
                    ),
                ]

            logging.debug(
                f"{split} dataset contains of %d positive and %d negative edges. \n Ratio = %f.",
                self.data.split_edge[f"{split}"]["pos"].shape[1],
                self.data.split_edge[f"{split}"]["neg"].shape[1],
                self.data.split_edge[f"{split}"]["neg"].shape[1]
                / self.data.split_edge[f"{split}"]["pos"].shape[1],
            )

        # Add opposite direction of positive training edges to make it undirected
        self.data.split_edge["train"]["pos"] = to_undirected(
            self.data.split_edge["train"]["pos"]
        )

        # Remove test and validation edges from edges list provided to SEAL model
        train_edge_boolean = []
        for i in edge_split_label:
            if i in (1, 2):  # If in valid or test split
                train_edge_boolean.append(False)
            else:  # If in train or no split
                train_edge_boolean.append(True)

        all_edges_except_test_and_valid = (
            self.data.pos_edge[:, train_edge_boolean].detach().clone()
        )
        self.data.edge_index = all_edges_except_test_and_valid

        # Store data objects
        torch.save(self.collate([self.data]), self.processed_paths[0])
