
import torch
import pandas as pd


def hitsK(y_true, y_pred, k):
    """Calculate the Hits@K. Based on definition in:
        M. Ali et al., "Bringing Light Into the Dark: A Large-scale Evaluation of Knowledge Graph
        Embedding Models under a Unified Framework"
        doi: 10.1109/TPAMI.2021.3124805.

    Args:
        y_true (1D tensor): Tensor with true labels.
        y_pred (1D tensor): Tensor wit predictions.
        'create_all_corrupted_df'.
        k (int): Calculate Hits Based on the top k prediction.

    Return:
        hits_k_score (float): Hits@K score
    """

    y_pred_neg = torch.tensor(y_pred[y_true == 0].tolist())
    y_pred_pos = torch.tensor(y_pred[y_true == 1].tolist())

    top_k_prob, _ = torch.topk(y_pred_neg, k)
    k_prob = float(top_k_prob[-1])

    hits_k_score = float(torch.sum(y_pred_pos > k_prob) / len(y_pred_pos))

    return hits_k_score


def mean_average_precision(y_true, y_pred, edges):
    """Calculate the mean of the average precision score. Based on definition in:
        M. Ali et al., "Bringing Light Into the Dark: A Large-scale Evaluation of Knowledge Graph
        Embedding Models under a Unified Framework"
        doi: 10.1109/TPAMI.2021.3124805.

    Args:
        y_true (iterable): Ground truth for each edge.
        y_pred (iterable): Prediction for each edge.
        edges (2d tensor or tuple wit 2 !D tensor): Test edges.

    Return:
        mean_ap (float): Mean average precision score.
        n_valid_nodes (int): Number of nodes included in the mean.
        len(set_nodes): Total number of unique nodes in the edges.
    """
    if len(edges) == 2:
        node_1, node_2 = edges[0], edges[1]
    else:
        node_1 = [e[0] for e in edges]
        node_2 = [e[1] for e in edges]

    df = pd.DataFrame(
        {"y true": y_true, "y pred": y_pred, "node 1": node_1, "node 2": node_2}
    )

    df = df.sort_values(by="y pred", ascending=False, ignore_index=True)

    all_nodes = [int(n) for n in node_1]
    all_nodes.extend([int(n) for n in node_2])

    set_nodes = [int(n) for n in set(all_nodes)]
    mean_ap = 0
    ap = 0
    count_included_nodes = 0
    for n in set_nodes:
        df_tmp = df[df["node 1"] == n].append(df[df["node 2"] == n]).drop_duplicates()
        df_tmp = df_tmp.sort_index()

        y_true = df_tmp["y true"].values
        if 0 in y_true and 1 in y_true:
            index_true = df_tmp[df_tmp["y true"] == 1].index
            for i in index_true:
                ap = 0
                df_tmp_k = df_tmp.loc[:i]
                ap += len(df_tmp_k[df_tmp_k["y true"] == 1].values) / len(
                    df_tmp_k["y true"].values
                )
            mean_ap += ap
        else:
            count_included_nodes += 1

    n_valid_nodes = len(set_nodes) - count_included_nodes
    if n_valid_nodes > 0:
        mean_ap /= n_valid_nodes
    else:
        mean_ap = None

    return mean_ap, n_valid_nodes, len(set_nodes)

