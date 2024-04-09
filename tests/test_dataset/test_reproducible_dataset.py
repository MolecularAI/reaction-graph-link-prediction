import pytest
import torch

from torch_trainer import GraphTrainer


@pytest.fixture
def settings():
    settings_dict = {
        "name": "test",
        "graph_path": "",  # TODO add graph path
        "train_fraction": 1,
        "splitting": "random",
        "mode": "normal",
        "valid_fold": 1,
        "neg_pos_ratio": 1,
        "neg_pos_ratio_test": 1,
        "fraction_dist_neg": 0,
        "use_attribute": True,
        "p_threshold": 0.5,
        "n_runs": 1,
        "seed": 1,
        "pos_weight_loss": 1,
    }

    return settings_dict


def test_variable_seed_sampling(settings):

    s = settings
    # s.sampling_factor = 1
    # s.fraction_dost_neg = 0.5
    print("settings", type(s), s)

    all_rand_neg_edges = []
    all_dist_neg_edges = []
    for i in range(4):
        trainer = GraphTrainer(settings)
        eln = trainer.initialize_data(i)

        all_rand_neg_edges.append(eln.data.neg_edge_rand)
        all_dist_neg_edges.append(eln.data.neg_edge_dist)

    all_rand_neg_edges = torch.cat(all_rand_neg_edges, dim=1)
    all_dist_neg_edges = torch.cat(all_dist_neg_edges, dim=1)

    if not eln.data.neg_edge_rand.shape[1] == 0:
        assert eln.data.neg_edge_rand.shape < all_rand_neg_edges.unique(dim=1).shape

    if not eln.data.neg_edge_dist.shape[1] == 0:
        assert eln.data.neg_edge_dist.shape < all_dist_neg_edges.unique(dim=1).shape


def test_fixed_seed_sampling(settings):

    # s = settings
    # s.sampling_factor = 1
    # s.fraction_dost_neg = 0.5

    all_rand_neg_edges = []
    all_dist_neg_edges = []
    for i in range(4):
        trainer = GraphTrainer(settings)
        eln = trainer.initialize_data(1)

        all_rand_neg_edges.append(eln.data.neg_edge_rand)
        all_dist_neg_edges.append(eln.data.neg_edge_dist)

    all_rand_neg_edges = torch.cat(all_rand_neg_edges, dim=1)
    all_dist_neg_edges = torch.cat(all_dist_neg_edges, dim=1)

    if not eln.data.neg_edge_rand.shape[1] == 0:
        assert eln.data.neg_edge_rand.shape == all_rand_neg_edges.unique(dim=1).shape

    if not eln.data.neg_edge_dist.shape[1] == 0:
        assert eln.data.neg_edge_dist.shape == all_dist_neg_edges.unique(dim=1).shape


def test_seed_init_model():
    pass

