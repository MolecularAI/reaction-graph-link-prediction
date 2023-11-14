import warnings
import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import graph_tool.all as gt


def get_molecule_degree(graph_path, save_file):

    graph = gt.load_graph(graph_path)

    molecules = gt.find_vertex(graph, graph.vertex_properties["labels"], ":Molecule")
    molecules_smiles = [graph.vertex_properties["smiles"][m] for m in molecules]
    molecules_out_degree = graph.get_total_degrees(graph.get_vertices())

    df = pd.DataFrame(
        {"smiles": molecules_smiles, "total degree": molecules_out_degree}
    )
    df = df.sort_values(by="total degree", ascending=False)

    df.to_csv(save_file)


save_file = "smiles_total_degree.csv"
graph_path = "graphs/monopartite/molecule_with_product_graph_5Fold.gt"

get_molecule_degree(graph_path, save_file)
