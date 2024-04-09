
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import graph_tool.all as gt


def get_reactants_and_product_index(graph):
    """Returns one list with the reactant molecule nodes and one list with the corresponding
    products molecule nodes.
    """
    reactant_index = []
    product_index = []

    reaction_class = []
    class_id = []

    reactions = gt.find_vertex(graph, graph.vertex_properties["labels"], ":Reaction")

    for r in reactions:
        in_neighbors = r.in_neighbors()
        in_neighbors = [int(n) for n in in_neighbors]
        if len(in_neighbors) != 0:
            reactant_index.append(in_neighbors)

            out_neighbors = r.out_neighbors()
            out_neighbors = [int(n) for n in out_neighbors]
            product_index.append(out_neighbors)

            try:
                reaction_class.append(graph.vertex_properties["reaction_class"][r])
                class_id.append(graph.vertex_properties["class_id"][r])
            except:
                reaction_class.append(None)
                class_id.append(None)

    return reactant_index, product_index, reaction_class, class_id


def get_smiles_to_index_dict(graph):

    smiles_to_index = {}
    for v in graph.get_vertices():
        smiles = graph.vertex_properties["smiles"][v]
        smiles_to_index[smiles] = v

    return smiles_to_index


def get_index_to_smiles_dict(graph):
    index_to_smiles = {}

    for v in graph.get_vertices():
        smiles = graph.vertex_properties["smiles"][v]
        index_to_smiles[v] = smiles

    return index_to_smiles


def shortest_distance(graph, edges):
    distances = []
    count = 0
    for node_1, node_2 in edges:
        count += 1
        dist = shortest_distance(
            graph, source=int(node_1), target=int(node_2), max_dist=50, directed=False
        )
        if dist == 2147483647:
            distances.append(50)
        else:
            distances.append(dist)
    return distances


def shortest_distance_positive(graph, edges):
    distances = []
    count = 0

    for node1, node2 in edges:
        count += 1
        graph.remove_edge(graph.edge(node1, node2))
        dist = shortest_distance(
            graph, source=int(node1), target=int(node2), max_dist=50, directed=False
        )
        graph.add_edge(node1, node2)
        if dist == 2147483647:
            distances.append(50)
        else:
            distances.append(dist)
    return distances

