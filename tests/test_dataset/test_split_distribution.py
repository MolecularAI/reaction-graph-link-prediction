import pytest
import torch

from datasets.reaction_graph import ReactionGraph
from datasets.seal import SEALDynamicDataset
from torch_trainer import GraphTrainer


@pytest.fixture
def settings():
    settings_dict = {
        "name": "test",
        "graph_path": "", # TODO add graph path
        "train_fraction": 1,
        "splitting": "random",
        "mode": "normal",
        "valid_fold": 1,
        "neg_pos_ratio": 1,
        "neg_pos_ratio_test": 1,
        "fraction_dist_neg": 0.5,
        "use_attribute": True,
        "p_threshold": 0.5,
        "n_runs": 1,
        "seed": 1,
        "pos_weight_loss": 1,
    }

    return settings_dict


def test_negative_sampling_distributions(settings):

    all_rand_neg_edges = []
    all_dist_neg_edges = []

    for fraction in (0, 0.5, 1):
        settings["fraction_dist_neg"] = fraction

        trainer = GraphTrainer(settings)
        eln = trainer.initialize_data(42)

        all_rand_neg_edges = eln.data.neg_edge_rand
        all_dist_neg_edges = eln.data.neg_edge_dist

        train_neg = eln.data.split_edge["train"]["neg"]
        valid_neg = eln.data.split_edge["valid"]["neg"]
        test_neg = eln.data.split_edge["test"]["neg"]

        train_and_rand = torch.cat((train_neg, all_rand_neg_edges), dim=1)
        unique_train_and_rand = train_and_rand.unique(dim=1)

        valid_and_rand = torch.cat((valid_neg, all_rand_neg_edges), dim=1)
        unique_valid_and_rand = valid_and_rand.unique(dim=1)

        test_and_rand = torch.cat((test_neg, all_rand_neg_edges), dim=1)
        unique_test_and_rand = test_and_rand.unique(dim=1)

        train_and_dist = torch.cat((train_neg, all_dist_neg_edges), dim=1)
        unique_train_and_dist = train_and_dist.unique(dim=1)

        valid_and_dist = torch.cat((valid_neg, all_dist_neg_edges), dim=1)
        unique_valid_and_dist = valid_and_dist.unique(dim=1)

        test_and_dist = torch.cat((test_neg, all_dist_neg_edges), dim=1)
        unique_test_and_dist = test_and_dist.unique(dim=1)

        if fraction == 0:
            assert len(unique_train_and_rand[0]) == len(all_rand_neg_edges[0])
            assert len(unique_valid_and_rand[0]) == len(all_rand_neg_edges[0])
            assert len(unique_test_and_rand[0]) == len(all_rand_neg_edges[0])
            assert len(unique_train_and_dist[0]) == len(all_dist_neg_edges[0]) + len(
                train_neg[0]
            )
            assert len(unique_valid_and_dist[0]) == len(all_dist_neg_edges[0]) + len(
                valid_neg[0]
            )
            assert len(unique_test_and_dist[0]) == len(all_dist_neg_edges[0]) + len(
                test_neg[0]
            )

        if fraction == 1:
            assert len(unique_train_and_dist[0]) == len(all_dist_neg_edges[0])
            assert len(unique_valid_and_dist[0]) == len(all_dist_neg_edges[0])
            assert len(unique_test_and_dist[0]) == len(all_dist_neg_edges[0])
            assert len(unique_train_and_rand[0]) == len(all_rand_neg_edges[0]) + len(
                train_neg[0]
            )
            assert len(unique_valid_and_rand[0]) == len(all_rand_neg_edges[0]) + len(
                valid_neg[0]
            )
            assert len(unique_test_and_rand[0]) == len(all_rand_neg_edges[0]) + len(
                test_neg[0]
            )

        if fraction == 0.5:
            assert (
                len(unique_train_and_dist[0])
                > len(all_dist_neg_edges[0]) + (len(train_neg[0]) / 2) * 0.9
            )
            assert (
                len(unique_valid_and_dist[0])
                > len(all_dist_neg_edges[0]) + (len(valid_neg[0]) / 2) * 0.9
            )
            assert (
                len(unique_test_and_dist[0])
                > len(all_dist_neg_edges[0]) + (len(test_neg[0]) / 2) * 0.9
            )
            assert (
                len(unique_train_and_rand[0])
                < len(all_rand_neg_edges[0]) + (len(train_neg[0]) / 2) * 1.1
            )
            assert (
                len(unique_valid_and_rand[0])
                < len(all_rand_neg_edges[0]) + (len(valid_neg[0]) / 2) * 1.1
            )
            assert (
                len(unique_test_and_rand[0])
                < len(all_rand_neg_edges[0]) + (len(test_neg[0]) / 2) * 1.1
            )
