from datetime import datetime
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt
import os.path as osp
import sys

from torch_trainer import GraphTrainer
from utils.reactions_info import get_index_to_smiles_dict


def predict_links(model_dir, edges, graph_path="", num_workers=None):
    """Reload pre-trained model and test with specified links.
    Args:
        model_dir (str): pathway to directory where the trained model is saved
        edges (tensor): tensor with dim (2, n) of edges that will be tested
    """

    # model dir path provided
    if osp.isfile(f"{model_dir}/settings.csv"):
        settings = pd.read_csv(f"{model_dir}/settings.csv")
    # model path provided
    elif osp.isfile(f'{"/".join(model_dir.split("/")[:-1])}/settings.csv'):
        settings = pd.read_csv(f'{"/".join(model_dir.split("/")[:-1])}/settings.csv')
    else:
        print("Wrong model_dir path provided:", model_dir)
        sys.exit()
    converted = []
    for i in settings["Settings"]:
        try:
            i_ = eval(i)
        except:
            i_ = i
        converted.append(i_)
    settings["value"] = converted
    settings = dict(zip(settings["Unnamed: 0"], settings["value"]))

    # Set path to pre-trained model and name
    settings["pre_trained_model_path"] = model_dir
    settings["name"] = f"evaluate_model/{model_dir}"
    # Reset parameter so that test is not set to empty when doing make_data_split
    settings["include_in_train"] = None

    if num_workers != None:
        settings["num_workers"] = num_workers
    if graph_path:
        settings["graph_path"] = graph_path

    if not "seed" in settings.keys():
        settings["seed"] = 1

    trainer = GraphTrainer(settings)
    eln_dataset = trainer.initialize_data(1)  # settings['seed'])

    if edges is not None:
        # assign evenly as positive and negative edges to test set.
        # The labels has no importance in themselves here, but the order does.
        n_edges = edges.size(1)
        eln_dataset.data.split_edge["test"]["pos"] = edges[:, : int(n_edges / 2)]
        eln_dataset.data.split_edge["test"]["neg"] = edges[:, int(n_edges / 2) :]

    datasets, dataloaders = trainer.make_data_splits(eln_dataset)
    model, _, _, _ = trainer.initialize_model(datasets)

    y_prob, links = trainer.predict(
        model, datasets, dataloaders, "test", None
    )

    return y_prob, links
