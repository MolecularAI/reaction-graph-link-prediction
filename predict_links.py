import argparse
import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt
from rdkit import Chem
from rdkit.Chem import Draw
import sys

# from torch_trainer import GraphTrainer
from utils.reactions_info import get_index_to_smiles_dict, get_neo4j_to_index_dict
from utils.evaluate_model import predict_links

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluating trained model")

parser.add_argument(
    "-p",
    "--model_dir_path",
    type=str,
)
parser.add_argument(
    "-s", "--save_path", type=str, default="predictions_current_best.csv"
)
parser.add_argument("-e", "--edges_path", type=str, default=None)
parser.add_argument("-g", "--graph_path", type=str, default="")
parser.add_argument(
    "-t", "--train_edges_path", type=str, default=None
)  # filename from data/TRAINING_MODEL_NAME/processed/processed_data.pt expected
parser.add_argument("-l", "--label", type=int, default=None)
parser.add_argument("-w", "--num_workers", type=int, default=None)

args = parser.parse_args()

model_dir_path = str(args.model_dir_path)
save_path = str(args.save_path)
edges_path = str(args.edges_path)
graph_path = str(args.graph_path)

if args.edges_path is not None:
    df_edges = pd.read_csv(edges_path)
    # edges = torch.tensor([df_edges['Reactant index 1'], df_edges['Reactant index 2']])

    source_col, target_col = "Source", "Target"  #'Reactant index 1', 'Reactant index 2'
    df_edges = df_edges.drop_duplicates(subset=[source_col, target_col])
    edges = torch.tensor([df_edges[source_col].values, df_edges[target_col].values])
    if args.label is not None:
        y_true = len(df_edges) * [args.label]
        df_edges["y true"] = y_true
    else:
        y_true = df_edges["y true"].values

    # check for overlap between edges and edges used for training the model and remove
    # NOTE: does not include the positive edges from the trained model in the check
    if args.train_edges_path is not None:
        data, _ = torch.load(args.train_edges_path)
        train_model_edges = (
            torch.cat((data.neg_edge_rand, data.neg_edge_dist), dim=1)
            .int()
            .t()
            .tolist()
        )
        train_model_edges_reversed = [edge[::-1] for edge in train_model_edges]
        bidirectional_train_model_edges = train_model_edges + train_model_edges_reversed
        edges_list = edges.t().tolist()

        filtered_edges_list, filtered_y_true = [], []
        for edge, y in zip(edges_list, y_true):
            if edge not in bidirectional_train_model_edges:
                filtered_edges_list.append(edge)
                filtered_y_true.append(y)
        edges = torch.tensor(filtered_edges_list).t()
        overlap_size = len(edges_list) - len(filtered_edges_list)
        print(
            f"{overlap_size} overlaps between edge list and train edges found and removed."
        )
        if len(filtered_edges_list) == 0:
            sys.exit("All edges overlap with train edges!")
        elif overlap_size > 0:
            df_edges = pd.DataFrame(
                data=edges.t().tolist(), columns=[source_col, target_col]
            )
            df_edges["y true"] = filtered_y_true

# Evaluate the model
print("Prediction started!")
y_prob, edges = predict_links(model_dir_path, edges=edges, graph_path=graph_path)
print("Prediction done!")

df_edges["y prob"] = y_prob
df_edges["edge"] = edges

df_edges.to_csv(save_path)
