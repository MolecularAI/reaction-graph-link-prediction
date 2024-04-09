
import torch
from torch_geometric.data import Dataset


class GeneralDataset(Dataset):
    def __init__(self, root, dataset, settings, split, **kwargs):
        self.data = dataset.data
        self.sampling_factor = settings["neg_pos_ratio"]
        self.split = split
        super(GeneralDataset, self).__init__(root)

        # Get positive and neative edges for given split
        self.pos_edge = self.data.split_edge[self.split]["pos"]
        self.neg_edge = self.data.split_edge[self.split]["neg"]

        # Creates a torch with all edges
        self.links = torch.cat([self.pos_edge, self.neg_edge], 1).t().tolist()
        self.labels = [1] * self.pos_edge.size(1) + [0] * self.neg_edge.size(1)

        edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)

    @property
    def num_features(self):
        if self.data.x != None:
            return len(self.data.x[0])
        else:
            return None

    def __len__(self):
        return len(self.links)

    def get(self, idx):

        return self.links[idx], self.labels[idx]

