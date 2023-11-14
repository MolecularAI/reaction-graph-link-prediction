import pytest

from torch_trainer import GraphTrainer
import os
import torch
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import graph_tool.all as gt

def test_folds():
    splittings = ["random"]  # , 'time']
    fractions = [0.1, 0.5, 1]  # [0, 0.5, 1]
    valid_fold = [1, 2, 3, 4, 5]

    for sp in splittings:
        for f in fractions:
            train_pos_vfs = []
            train_neg_vfs = []
            valid_pos_vfs = []
            valid_neg_vfs = []
            test_pos_vfs = []
            test_neg_vfs = []

            for vf in valid_fold:
                print("split", sp, "fraction", f, "valid_fold", vf)

                settings = {
                    "name": f"test_split_overlap_{sp}_{f}_{vf}",
                    "graph_path": "",  # TODO add graph path
                    "train_fraction": f,
                    "splitting": sp,
                    "valid_fold": vf,
                    "percent_edges": 2,
                    "fraction_dist_neg": 0,
                    "neg_pos_ratio": f,
                    "neg_pos_ratio_test": 1,
                    "use_attribute": False,
                    "p_threshold": 0.5,
                    "mode": "normal",
                    "n_runs": 1,
                    "seed": 1,
                    "pos_weight_loss": 1,
                }

                os.system(f"rm -rf data/{settings['name']}")
                trainer = GraphTrainer(settings)
                eln = trainer.initialize_data(settings["seed"])

                train_pos_vfs.append(eln.data.split_edge["train"]["pos"])
                train_neg_vfs.append(eln.data.split_edge["train"]["neg"])
                valid_pos_vfs.append(eln.data.split_edge["valid"]["pos"])
                valid_neg_vfs.append(eln.data.split_edge["valid"]["neg"])
                test_pos_vfs.append(eln.data.split_edge["test"]["pos"])
                test_neg_vfs.append(eln.data.split_edge["test"]["neg"])

                all_pos = torch.cat(
                    (
                        eln.data.split_edge["train"]["pos"],
                        eln.data.split_edge["valid"]["pos"],
                        eln.data.split_edge["test"]["pos"],
                    ),
                    dim=1,
                )
                all_neg = torch.cat(
                    (
                        eln.data.split_edge["train"]["neg"],
                        eln.data.split_edge["valid"]["neg"],
                        eln.data.split_edge["test"]["neg"],
                    ),
                    dim=1,
                )
                all_edges = torch.cat((all_pos, all_neg), dim=1)

                assert all_pos.unique(dim=1).shape == all_pos.shape, (
                    all_pos.unique(dim=1).shape,
                    all_pos.shape,
                )
                assert all_neg.unique(dim=1).shape == all_neg.shape, (
                    all_neg.unique(dim=1).shape,
                    all_neg.shape,
                )
                assert all_edges.unique(dim=1).shape == all_edges.shape, (
                    all_edges.unique(dim=1).shape,
                    all_edges.shape,
                )

            assert (
                torch.cat(valid_pos_vfs, dim=1).unique(dim=1).shape
                == torch.cat(valid_pos_vfs, dim=1).shape
            )
            assert torch.cat(test_pos_vfs, dim=1).unique(dim=1).shape[1] == int(
                torch.cat(test_pos_vfs, dim=1).shape[1] / len(valid_fold)
            )
            assert (
                torch.cat(valid_neg_vfs, dim=1).unique(dim=1).shape
                == torch.cat(valid_neg_vfs, dim=1).shape
            )
            assert torch.cat(test_neg_vfs, dim=1).unique(dim=1).shape[1] == int(
                torch.cat(test_neg_vfs, dim=1).shape[1] / len(valid_fold)
            )
