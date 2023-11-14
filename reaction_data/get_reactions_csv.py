import argparse
import sys

import warnings
import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import graph_tool.all as gt
from utils.reactions_info import (
    get_reactants_and_product_index,
    shortest_distance_positive,
)
from utils.reactions_info import get_index_to_smiles_dict


def get_reactions_csv(args):
    graph = gt.load_graph(args.graph_bipartite)
    (
        reactant_index,
        product_index,
        reaction_class,
        class_id,
    ) = get_reactants_and_product_index(graph)
    index_to_smiles = get_index_to_smiles_dict(graph)

    # Check for non-binary reactions
    for i, j in zip(reactant_index, product_index):
        assert len(i) == 2
        assert len(j) == 1

    # Edges
    edges = np.array(reactant_index)[:, 0], np.array(reactant_index)[:, 1]

    # Create dataframe with all reactions
    reactant_products_df = pd.DataFrame(
        {
            "Reactant index 1": edges[0],
            "Reactant index 2": edges[1],
            "Product index": np.array(product_index)[:, 0],
            "Reactant smiles 1": [index_to_smiles[i] for i in edges[0]],
            "Reactant smiles 2": [index_to_smiles[i] for i in edges[1]],
            "Product smiles": [
                index_to_smiles[i] for i in np.array(product_index)[:, 0]
            ],
        }
    )
    # Save dataframe
    reactant_products_df.to_csv(args.save_file_name, index=False)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_bipartite", type=str, default=None)
    parser.add_argument("-s", "--save_file_name", type=str, default=None)
    args = parser.parse_args()

    get_reactions_csv(args)
