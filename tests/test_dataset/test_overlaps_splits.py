import pytest

from torch_trainer import GraphTrainer
import os
import torch
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import graph_tool.all as gt


def test_split_overlaps():
    splittings = ["random"] 
    fractions = [0]
    seeds = [1, 2]

    for sp in splittings:
        for f in fractions:
            for se in seeds:
                print("split", sp, "fraction", f, "seed", se)

                settings = {
                    "name": f"test_split_overlap_{sp}_{f}_{f}",
                    "graph_path": "",  # TODO add graph path
                    "train_fraction": 1,
                    "splitting": sp,
                    "valid_fold": 1,
                    "percent_edges": 2,
                    "fraction_dist_neg": f,
                    "neg_pos_ratio": 1,
                    "neg_pos_ratio_test": 1,
                    "use_attribute": False,
                    "p_threshold": 0.5,
                    "mode": "normal",
                    "n_runs": 1,
                    "seed": se,
                    "pos_weight_loss": 1,
                }

                os.system(f"rm -rf data/{settings['name']}")
                trainer = GraphTrainer(settings)
                eln = trainer.initialize_data(se)

                train_pos = eln.data.split_edge["train"]["pos"]
                train_neg = eln.data.split_edge["train"]["neg"]
                valid_pos = eln.data.split_edge["valid"]["pos"]
                valid_neg = eln.data.split_edge["valid"]["neg"]
                test_pos = eln.data.split_edge["test"]["pos"]
                test_neg = eln.data.split_edge["test"]["neg"]

                all_edges = torch.cat(
                    (train_pos, train_neg, valid_pos, valid_neg, test_pos, test_neg),
                    dim=1,
                )
                all_train = torch.cat((train_pos, train_neg), dim=1)
                all_test = torch.cat((test_pos, test_neg), dim=1)
                all_valid = torch.cat((valid_pos, valid_neg), dim=1)
                all_neg = torch.cat((train_neg, valid_neg, test_neg), dim=1)
                all_pos = torch.cat((train_pos, valid_pos, test_pos), dim=1)

                assert all_edges.shape == all_edges.unique(dim=0).shape
                assert all_train.shape == all_train.unique(dim=0).shape
                assert all_valid.shape == all_valid.unique(dim=0).shape
                assert all_test.shape == all_test.unique(dim=0).shape
                assert all_neg.shape == all_neg.unique(dim=0).shape
                assert all_pos.shape == all_pos.unique(dim=0).shape
