import argparse
from datetime import datetime
import sys
import warnings

import itertools
import numpy as np
import pandas as pd
import torch
import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import graph_tool.all as gt
from graph_tool.topology import shortest_distance

from datasets.reaction_graph import ReactionGraph
from datasets.seal import SEALDynamicDataset
from settings import settings
from utils.negative_sampling import (
    one_against_all,
    one_against_most_reactive,
)
from utils.reactions_info import get_index_to_smiles_dict

from itertools import combinations

# -------------------------------- Functions --------------------------------


def sample_negative_one_against_all(
    fixed_reactant_path, graph_path, neg_pos_ratio, most_reactive_only=False
):

    # Load Graph
    graph = gt.load_graph(graph_path)

    # Load settings
    s = settings.settings
    s["graph_path"] = graph_path
    s["train_fraction"] = 1
    s["neg_pos_ratio_test"] = 1
    s["neg_pos_ratio"] = neg_pos_ratio
    s["fraction_dist_neg"] = 0
    s["seed"] = 1
    s["name"] = f"get_info_{datetime.now()}"
    s["valid_fold"] = 4

    # Create ReactionGraph dataset
    eln = ReactionGraph(
        f"data/get_info_{datetime.now()}", settings.settings, seed_neg_sampling=0
    )

    i_pos, j_pos = eln.data.pos_edge
    print("num links", len(i_pos))
    print(max(i_pos), max(j_pos))

    fixed_reactant_df = pd.read_csv(fixed_reactant_path)
    fixed_reactant_idx = fixed_reactant_df["node index in graph"].values

    if most_reactive_only:
        i_neg, j_neg = one_against_most_reactive(
            fixed_reactant_idx, (i_pos, j_pos), all_pos_edges=(i_pos, j_pos), cutoff=2
        )
    else:
        i_neg, j_neg = one_against_all(
            fixed_reactant_idx,
            (i_pos, j_pos),
            all_pos_edges=(i_pos, j_pos),
            include_unconnected=False,
        )

    # Create dictionaries, from gt index to neo4j index and from gt index to smiles
    index_smils_dict = get_index_to_smiles_dict(graph)

    for s, i in zip(
        fixed_reactant_df["SMILES"].values,
        fixed_reactant_df["node index in graph"].values,
    ):
        index_smils_dict[i] = s

    # Negative edges sampled at random
    neg_edge = eln.data.neg_edge_rand

    neg_edge_idx_1 = i_neg.to(int)
    neg_edge_idx_2 = j_neg.to(int)
    neg_edge_smiles_1 = [index_smils_dict[int(i)] for i in neg_edge_idx_1]
    neg_edge_smiles_2 = [index_smils_dict[int(i)] for i in neg_edge_idx_2]

    neg_reactions_df = pd.DataFrame(
        {
            "Source": neg_edge_idx_1,
            "Target": neg_edge_idx_2,
            "Reactant smiles 1": neg_edge_smiles_1,
            "Reactant smiles 2": neg_edge_smiles_2,
            "Type": ["all" for i in range(len(neg_edge_idx_1))],
            "y true": 0,
        }
    )
    return neg_reactions_df, graph


def sample_negative_reactions(graph_path, neg_pos_ratio, fraction_dist_neg, seed=1000):

    # Load Graph
    graph = gt.load_graph(graph_path)

    # Load settings
    s = settings.settings
    s["graph_path"] = graph_path
    s["train_fraction"] = 1
    s["neg_pos_ratio_test"] = 1
    s["neg_pos_ratio"] = neg_pos_ratio
    s["fraction_dist_neg"] = fraction_dist_neg
    s["valid_fold"] = 0

    # Create ReactionGraph dataset
    eln_dataset = ReactionGraph(
        f"data/get_info_{datetime.now()}", settings.settings, seed_neg_sampling=seed
    )

    # Create dictionaries, from gt index to neo4j index and from gt index to smiles
    index_smils_dict = get_index_to_smiles_dict(graph)

    # Negative edges sampled at random
    neg_edge_rand = eln_dataset.data.neg_edge_rand

    neg_edge_rand_idx_1 = neg_edge_rand[0, :].to(int)
    neg_edge_rand_idx_2 = neg_edge_rand[1, :].to(int)
    neg_edge_rand_smiles_1 = [index_smils_dict[int(i)] for i in neg_edge_rand_idx_1]
    neg_edge_rand_smiles_2 = [index_smils_dict[int(i)] for i in neg_edge_rand_idx_2]

    neg_edge_rand_smiles_df = pd.DataFrame(
        {
            "Reactant index 1": neg_edge_rand_idx_1,
            "Reactant index 2": neg_edge_rand_idx_2,
            "Reactant smiles 1": neg_edge_rand_smiles_1,
            "Reactant smiles 2": neg_edge_rand_smiles_2,
            "Type": ["random" for i in range(len(neg_edge_rand_idx_1))],
        }
    )

    # Negative edges sampled from positive edges distribution
    neg_edge_dist = eln_dataset.data.neg_edge_dist
    neg_edge_dist_idx_1 = neg_edge_dist[0, :].to(int)
    neg_edge_dist_idx_2 = neg_edge_dist[1, :].to(int)
    neg_edge_dist_smiles_1 = [index_smils_dict[int(i)] for i in neg_edge_dist_idx_1]
    neg_edge_dist_smiles_2 = [index_smils_dict[int(i)] for i in neg_edge_dist_idx_2]

    neg_edge_dist_smiles_df = pd.DataFrame(
        {
            "Reactant index 1": neg_edge_dist_idx_1,
            "Reactant index 2": neg_edge_dist_idx_2,
            "Reactant smiles 1": neg_edge_dist_smiles_1,
            "Reactant smiles 2": neg_edge_dist_smiles_2,
            "Type": ["distribution" for i in range(len(neg_edge_dist_idx_1))],
        }
    )

    # Append the two dataframes
    neg_reactions_df = neg_edge_rand_smiles_df.append(
        neg_edge_dist_smiles_df, ignore_index=True
    )
    return neg_reactions_df, graph


# -------------------------------- Main --------------------------------


def main(args):

    if args.sample_how == "fixed_reactant":
        neg_reactions_df, _ = sample_negative_one_against_all(
            args.fixed_reactant_path,
            args.graph_path,
            args.neg_pos_ratio,
            most_reactive_only=False,
        )
    elif args.sample_how == "most_reactive":
        neg_reactions_df, _ = sample_negative_one_against_all(
            args.fixed_reactant_path,
            args.graph_path,
            args.neg_pos_ratio,
            most_reactive_only=True,
        )
    else:
        neg_reactions_df, _ = sample_negative_reactions(
            args.graph_path,
            args.neg_pos_ratio,
            args.fraction_dist_neg,
            args.seed,
        )

    neg_reactions_df.to_csv(args.save_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--graph_path", type=str)
    parser.add_argument("--graph_with_templates_path", type=str)
    parser.add_argument("--sample_how", type=str, default="normal")
    parser.add_argument("--fixed_reactant_path", type=str, default="")
    parser.add_argument("--neg_pos_ratio", type=int, default=0.5)
    parser.add_argument("--fraction_dist_neg", type=float)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--reactive_functions_count", type=str)
    parser.add_argument("--count_cut_off", type=int, default=1)
    parser.add_argument("--save_file_path", type=str)

    args = parser.parse_args()
    main(args)
