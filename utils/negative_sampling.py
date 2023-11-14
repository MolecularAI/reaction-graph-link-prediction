import logging
import numpy as np
import torch
import warnings
from collections import Counter
from tqdm import tqdm
from sys import maxsize
from os.path import isfile

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import graph_tool.all as gt


def correct_overlaps(pos_edge_index, neg_edge_index, num_nodes, seed):
    """Checks for overlap between pos_edge_index and neg_edge_index and replaces these,
    while considering the edges as undirected.

    Args:
        pos_edge_index (2D tensor or tuple with 2 1D tesor): Used to compare neg_edge_index to
        neg_edge_index (2D tensor or tuple with 2 1D tesor): Used to check if any overlap with
            pos_edge_index
        num_nodes (int): Number of nodes in the graph.
        seed (int): Randoms seed used by torch.

    Returns:
        neg_edge_index_1 (tensor): neg_edge_index_1, same as input
        neg_edge_index_2 (tensor): Updated version of negative_edge_index so no overlaps
            exists with pos_edge_index.
    """

    torch.manual_seed(seed)
    pos_edge_index_1, pos_edge_index_2 = pos_edge_index
    neg_edge_index_1, neg_edge_index_2 = neg_edge_index

    all_nodes = torch.arange(num_nodes)

    # Creating idx_1 (unique identifier) for both edge directions
    idx_1 = torch.cat(
        (
            pos_edge_index_1 * num_nodes + pos_edge_index_2,
            pos_edge_index_2 * num_nodes + pos_edge_index_1,
            all_nodes * num_nodes + all_nodes,
        )
    )

    idx_2 = neg_edge_index_1 * num_nodes + neg_edge_index_2

    mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)

    logging.warning(
        "%d overlaps have been found. \n Comparing: %d edges with %d negative edges.",
        rest.numel(),
        pos_edge_index_1.size(0),
        neg_edge_index_1.size(0),
    )

    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.numel(),), dtype=torch.long)
        idx_2 = neg_edge_index_1[rest] * num_nodes + tmp
        mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
        neg_edge_index_2[rest] = tmp
        rest = rest[mask.nonzero(as_tuple=False).view(-1)]

    return neg_edge_index_1, neg_edge_index_2


def remove_overlaps(corrupted_edge_index, pos_edge_index):
    """Checks for overlap between corrupted_edge_index and pos_edge_index and removes these,
    while considering the edges as undirected.

    Args:
        pos_edge_index (2D tensor or tuple with 2 1D tesor): Used to compare neg_edge_index to
        corrupted_edge_index (2D tensor or tuple with 2 1D tesor): Used to check if any overlap with
            pos_edge_index

    Returns:
        neg_edge_index_1 (tensor): Updated version of corrupted_edge_index with any overlaps
            with pos_edge_index removed.
        neg_edge_index_2 (tensor): Updated version of corrupted_edge_index with any overlaps
            with pos_edge_index removed.
    """

    pos_edge_index_1, pos_edge_index_2 = pos_edge_index
    neg_edge_index_1, neg_edge_index_2 = corrupted_edge_index

    all_nodes = [int(i) for i in pos_edge_index_1]
    all_nodes.extend([int(i) for i in pos_edge_index_2])
    all_nodes = torch.tensor(list(set(all_nodes)))

    num_nodes = max(all_nodes)

    # Creating idx_1 (unique identifier) for both edge dirctions
    idx_1 = torch.cat(
        (
            pos_edge_index_1 * num_nodes + pos_edge_index_2,
            pos_edge_index_2 * num_nodes + pos_edge_index_1,
            all_nodes * num_nodes + all_nodes,
        )
    )

    idx_2 = neg_edge_index_1 * num_nodes + neg_edge_index_2

    mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)

    neg_edge_index_1 = neg_edge_index_1[~mask]
    neg_edge_index_2 = neg_edge_index_2[~mask]

    return neg_edge_index_1, neg_edge_index_2


# ---------------------------- Negative Sampling Methods ----------------------------
def sample_analogs(edge_index, n, num_nodes, all_pos_edges, seed):
    """Sample negative edges from the distribution of nodes in the positive edges
     where positive set node degrees are preserved

     Args:
         edge_index (2D tensor or tuple with 2 1D tesor): The edges to base the sampling on.
         n (int): How many edges to sample: n * number of edges in edge_index negative edges.
         num_nodes (int): Number of nodes in the graph.
         all_pos_edges: All edges in the graph.
         seed (int): Random seed for torch.

    Return:
         i_neg (1D tensors): Sampled negative edge sources.
         j_neg (1D tensors): Sampled negative edge targets.
    """

    torch.manual_seed(seed)

    # check if degree distribution already exists for a given seed
    negative_analogs_file = (
        "data/negatives=analogs_USPTO_1_fingerprints_single_NameRxn3.2.pt"
    )

    if isfile(negative_analogs_file):
        neg_edges = torch.load(negative_analogs_file)
        logging.info(
            "Loading pre-computed negative links of 2-nodes away nodes file %s",
            negative_analogs_file,
        )
    else:
        sys.exit()

    neg_edges = neg_edges[:, torch.randperm(neg_edges.shape[1])[:n]]

    if neg_edges.shape[1] < n:
        logging.warning(
            "To few unique sampled negative edges by sample_distribution. \
                        Fill with random edges."
        )
        # Fill with randomly sampled edges
        n_rand = n - neg_edges.shape[1]
        i_neg_rand, j_neg_rand = sample_random(
            edge_index, n_rand, int(1.5 * num_nodes), all_pos_edges, seed=seed
        )
        ij_rand = torch.stack((i_neg_rand, j_neg_rand), dim=0)
        # cat and remove repetetive edges - does not replace the removed ones!
        neg_edges = torch.cat((neg_edges, ij_rand), dim=1).unique(dim=1)

    i_neg, j_neg = neg_edges[:, :n]

    return i_neg, j_neg


def sample_degree_preserving_distribution(
    negative_degree_preserving_distribution_file,
    edge_index,
    n,
    num_nodes,
    all_pos_edges,
    seed,
):
    """Sample negative edges from the distribution of nodes in the positive edges
     where positive set node degrees are preserved

     Args:
         edge_index (2D tensor or tuple with 2 1D tesor): The edges to base the sampling on.
         n (int): How many edges to sample: n * number of edges in edge_index negative edges.
         num_nodes (int): Number of nodes in the graph.
         all_pos_edges: All edges in the graph.
         seed (int): Random seed for torch.

    Return:
         i_neg (1D tensors): Sampled negative edge sources.
         j_neg (1D tensors): Sampled negative edge targets.
    """

    torch.manual_seed(seed)

    # check if degree distribution already exists
    if isfile(negative_degree_preserving_distribution_file):
        neg_edges = torch.load(negative_degree_preserving_distribution_file)
        logging.info(
            "Loading pre-computed degree-preserving distribution negative links with a fixed seed %d from %s'.",
            seed,
            negative_degree_preserving_distribution_file,
        )
    else:
        print("Generating negative links preserving degree distribution. Take a seat.")

        i_pos, j_pos = edge_index
        ij = torch.cat((i_pos, j_pos))
        all_nodes_in_pos_edges = ij.tolist()

        # loop descending by node popularity to decrease chance of insufficient unique link pair nodes available for popular nodes
        # NOTE: this leads overesimated predicted duration of sampling by tqdm bar
        counter = Counter(all_nodes_in_pos_edges).most_common()

        source_nodes, target_nodes = [], []
        i = 0
        pbar = tqdm(total=len(counter))
        while i < len(counter):
            source = counter[i][0]
            degree = counter[i][1]
            target = maxsize

            # get positive head or tail partners of node in question
            j_partners = j_pos[i_pos == source]
            i_partners = i_pos[j_pos == source]

            # make list of forbidden partner nodes which includes partner nodes of node in question as well as itself
            nodes_for_exclusion = set(
                torch.cat((i_partners, j_partners, torch.tensor([source]))).tolist()
            )

            while degree > 0:
                all_nodes_in_pos_edges.remove(source)

                # add target nodes already used for source node in question
                nodes_for_exclusion.add(target)

                # make list for sampling which does not include any forbidden nodes
                nodes_for_sampling = [
                    item
                    for item in all_nodes_in_pos_edges
                    if item not in nodes_for_exclusion
                ]

                if len(nodes_for_sampling) > 0:
                    target = nodes_for_sampling[
                        torch.randint(0, len(nodes_for_sampling), size=(1,))
                    ]
                    all_nodes_in_pos_edges.remove(target)
                    source_nodes.append(source)
                    target_nodes.append(target)

                    # reduce counter for target node found
                    k = 0
                    while k < len(counter):
                        if counter[k][0] == target:
                            counter[k] = tuple([counter[k][0], counter[k][1] - 1])
                            break
                        k += 1
                degree -= 1
            i += 1
            pbar.update(1)
        pbar.close()

        i_neg = torch.tensor(source_nodes)
        j_neg = torch.tensor(target_nodes)

        i_neg, j_neg = correct_overlaps(all_pos_edges, (i_neg, j_neg), num_nodes, seed)
        neg_edges = torch.stack((i_neg, j_neg), dim=0)

        old_size = len(neg_edges[0])
        neg_edges = neg_edges.unique(dim=1)
        logging.info(
            "%d (%f) duplicate edges from distribution were removed.",
            old_size - len(neg_edges[0]),
            (old_size - len(neg_edges[0])) / old_size,
        )
        torch.save(neg_edges, negative_degree_preserving_distribution_file)

    neg_edges = neg_edges[:, torch.randperm(neg_edges.shape[1])[:n]]

    if neg_edges.shape[1] < n:
        logging.warning(
            "To few unique sampled negative edges by sample_distribution. \
                        Fill with random edges."
        )
        # Fill with randomly sampled edges
        n_rand = n - neg_edges.shape[1]
        i_neg_rand, j_neg_rand = sample_random(
            edge_index, n_rand, int(1.5 * num_nodes), all_pos_edges, seed=seed
        )
        ij_rand = torch.stack((i_neg_rand, j_neg_rand), dim=0)
        # cat and remove repetetive edges - does not replace the removed ones!
        neg_edges = torch.cat((neg_edges, ij_rand), dim=1).unique(dim=1)

    i_neg, j_neg = neg_edges[:, :n]

    return i_neg, j_neg


def sample_distribution(edge_index, n, num_nodes, all_pos_edges, seed):
    """Sample negative edges from the distribution of nodes in the positive edges.

     Args:
         edge_index (2D tensor or tuple with 2 1D tesor): The edges to base the sampling on.
         n (int): How many edges to sample: n * number of edges in edge_index negative edges.
         num_nodes (int): Number of nodes in the graph.
         all_pos_edges: All edges in the graph.
         seed (int): Random seed for torch.

    Return:
         i_neg (1D tensors): Sampled negative edge sources.
         j_neg (1D tensors): Sampled negative edge targets.
    """

    torch.manual_seed(seed)

    i_pos, j_pos = edge_index
    ij = torch.cat((i_pos, j_pos))

    source, target = [], []
    x = int(np.ceil(n / len(i_pos)))
    for _ in range(x + 1):
        ij = ij[torch.randperm(len(ij))]
        source.append(ij[0 : len(i_pos)])
        target.append(ij[len(i_pos) :])

    i_neg = torch.cat(source)
    j_neg = torch.cat(target)

    i_neg, j_neg = correct_overlaps(all_pos_edges, (i_neg, j_neg), num_nodes, seed)
    neg_edges = torch.stack((i_neg, j_neg), dim=0)
    old_size = len(neg_edges[0])
    neg_edges = neg_edges.unique(dim=1)
    logging.info(
        "%d (%f) duplicate edges from distribution were removed.",
        old_size - len(neg_edges[0]),
        (old_size - len(neg_edges[0])) / old_size,
    )
    neg_edges = neg_edges[:, torch.randperm(neg_edges.shape[1])[:n]]

    if neg_edges.shape[1] < n:
        logging.warning(
            "To few unique sampled negative edges by sample_distribution. \
                        Fill with random edges."
        )
        # Fill with randomly sampled edges
        n_rand = n - neg_edges.shape[1]
        i_neg_rand, j_neg_rand = sample_random(
            edge_index, n_rand, int(1.5 * num_nodes), all_pos_edges, seed=seed
        )
        ij_rand = torch.stack((i_neg_rand, j_neg_rand), dim=0)
        # cat and remove repetetive edges - does not replace the removed ones!
        neg_edges = torch.cat((neg_edges, ij_rand), dim=1).unique(dim=1)

    i_neg, j_neg = neg_edges[:, :n]

    return i_neg, j_neg


def sample_random(edge_index, n, num_nodes, all_pos_edges, seed):
    """Sample negative edges, both the source and target, at random.

     Args:
         edge_index (2D tensor or tuple with 2 1D tesor): The edges to base the sampling on.
         n (int): How many edges to sample: n * number of edges in edge_index negative edges.
         num_nodes (int): Number of nodes in the graph.
         all_pos_edges: All edges in the graph.
         seed (int): Random seed for torch.

    Return:
         i_neg (1D tensors): Sampled negative edge sources.
         j_neg (1D tensors): Sampled negative edge targets.
    """

    torch.manual_seed(seed)

    all_nodes = torch.cat((edge_index[0], edge_index[1])).flatten().unique()

    i_index = torch.randint(len(all_nodes), (int(1.5 * n),), dtype=torch.long)
    j_index = torch.randint(len(all_nodes), (int(1.5 * n),), dtype=torch.long)

    i = all_nodes[i_index]
    j = all_nodes[j_index]

    i, j = correct_overlaps(all_pos_edges, (i, j), num_nodes, seed)

    neg_edges = torch.stack((i, j), dim=0)

    neg_edges = neg_edges.unique(dim=1)

    neg_edges = neg_edges[:, torch.randperm(neg_edges.shape[1])[:n]]

    if neg_edges.shape[1] < n:
        logging.warning("To few unique sampled negative edges by sample_random.")
        i, j = neg_edges
    else:
        i, j = neg_edges[:, :n]

    return i, j


def one_against_all(nodes, edge_index, all_pos_edges, include_unconnected=False):
    """Sample negative edges, by keeping a fixed target node and sampling all possible source nodes.

     Args:
         nodes (iterable): Fixed reactant
         edge_index (2D tensor or tuple with 2 1D tesor): The edges to base the sampling on.
         all_pos_edges (2D tensor or tuple with 2 1D tesor): All edges in the graph.

    Return:
         neg_edges (2D tensors): Negative edges.
    """
    all_nodes_in_edges = torch.cat((edge_index[0], edge_index[1])).flatten().unique()
    if include_unconnected:
        all_nodes = torch.arange(1, max(all_nodes_in_edges))
    else:
        all_nodes = all_nodes_in_edges
    print("num unique nodes", len(all_nodes.unique()))
    i_neg = torch.tensor([])
    j_neg = torch.tensor([])
    for node in nodes:
        node = int(node)
        i_neg_node = all_nodes[all_nodes != node]
        j_neg_node = torch.tensor([node for _ in i_neg_node.flatten()])

        if node in all_nodes:
            i_neg_node, j_neg_node = remove_overlaps(
                (i_neg_node, j_neg_node), all_pos_edges
            )
        else:
            print(f"Node {node} not in reactants.")

        print("num unique nodes", len(all_nodes.unique()))
        i_neg = torch.cat((i_neg, i_neg_node))
        j_neg = torch.cat((j_neg, j_neg_node))

    neg_edges = torch.stack((i_neg, j_neg), dim=0)

    return neg_edges


def one_against_most_reactive(nodes, edge_index, all_pos_edges, cutoff=2):
    """Sample negative edges, by keeping a fixed target node and sampling all possible source nodes.

     Args:
         nodes (iterable): Fixed reactant
         edge_index (2D tensor or tuple with 2 1D tesor): The edges to base the sampling on.
         all_pos_edges (2D tensor or tuple with 2 1D tesor): All edges in the graph.

    Return:
         neg_edges (2D tensors): Negative edges.
    """
    all_nodes = torch.cat((edge_index[0], edge_index[1])).flatten().tolist()
    print("len(all_nodes)", len(all_nodes), "max ", max(all_nodes))

    count_reactants = Counter(all_nodes)
    reactive_reactants = [item for item in set(all_nodes) if count_reactants[item] > 5]
    reactive_reactants = torch.tensor(list(set(reactive_reactants)))
    print("num unique nodes", len(reactive_reactants))

    i_neg = torch.tensor([])
    j_neg = torch.tensor([])
    for node in nodes:
        node = int(node)
        i_neg_node = reactive_reactants[reactive_reactants != node]
        j_neg_node = torch.tensor([node for _ in i_neg_node.flatten()])

        if node in all_nodes:
            i_neg_node, j_neg_node = remove_overlaps(
                (i_neg_node, j_neg_node), all_pos_edges
            )
        else:
            print(f"Node {node} not in reactants.")

        # print('num unique nodes', len(reactive_reactants))
        i_neg = torch.cat((i_neg, i_neg_node))
        j_neg = torch.cat((j_neg, j_neg_node))

    neg_edges = torch.stack((i_neg, j_neg), dim=0)

    return neg_edges


def all_against_all(nodes, edge_index, all_pos_edges, include_unconnected=False):
    """Sample negative edges, by keeping a fixed target node and sampling all possible source nodes.

     Args:
         nodes (iterable): Fixed reactant
         edge_index (2D tensor or tuple with 2 1D tesor): The edges to base the sampling on.
         all_pos_edges (2D tensor or tuple with 2 1D tesor): All edges in the graph.

    Return:
         neg_edges (2D tensors): Negative edges.
    """
    all_nodes_in_edges = torch.cat((edge_index[0], edge_index[1])).flatten().unique()
    if include_unconnected:
        all_nodes = torch.arange(1, max(all_nodes_in_edges))
    else:
        all_nodes = all_nodes_in_edges
    print("num unique nodes", len(all_nodes.unique()))

    i_neg = torch.tensor([])
    j_neg = torch.tensor([])
    for node in nodes:
        node = int(node)
        i_neg_node = all_nodes[all_nodes != node]
        j_neg_node = torch.tensor([node for _ in i_neg_node.flatten()])

        if node in all_nodes:
            i_neg_node, j_neg_node = remove_overlaps(
                (i_neg_node, j_neg_node), all_pos_edges
            )
        else:
            print(f"Node {node} not in reactants.")

        print("num unique nodes", len(all_nodes.unique()))
        i_neg = torch.cat((i_neg, i_neg_node))
        j_neg = torch.cat((j_neg, j_neg_node))

    neg_edges = torch.stack((i_neg, j_neg), dim=0)

    return neg_edges
