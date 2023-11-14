import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score, roc_curve

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import graph_tool.all as gt


# get edges as edge list
def get_edges(edges_series):
    """From a panda series convert containing: '[id1, id2]' convert from edge to edge list"""

    edges = []
    for edge in edges_series:
        tmp = edge.strip("][")
        tmp = list(map(int, tmp.split(",")))
        edges.append(tmp)

    return edges


# get edges as edge list
def get_edges(edges_series):

    edges = []
    for edge in edges_series:
        tmp = edge.strip("][")
        tmp = list(map(int, tmp.split(",")))
        edges.append(tmp)
    return edges


def create_id_csv(graph, file_path):
    """Creates a csv file where each gt id is mapped to its neo4j id"""

    neo4j_id = []
    gt_id = []

    for v in graph.vertices():
        gt_id.append(int(v))
        neo4j_id.append(int(graph.vertex_properties["_graphml_vertex_id"][v][1:]))

    nodes_labels_dict = {"renamed_ID": gt_id, "neo4j_ID": neo4j_id}
    nodes_labels_dict = pd.DataFrame(nodes_labels_dict)

    if file_path:
        nodes_labels_dict.to_csv(file_path, index=False)
        print("Saved!")

    return nodes_labels_dict


# --------- Find Optimal Threshold ---------


def find_optimal_threshold(roc_curve, plot=True, title=None, save_name=None):
    """Find optimal threshold based on the roc curve."""

    fpr, tpr, threshold = roc_curve
    if plot:
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.plot(threshold, 1 - fpr, label="1 - FPR", zorder=1)
        ax.plot(threshold, tpr, label="TPR", zorder=2)

        where = np.where(np.round(1 - fpr, 2) == np.round(tpr, 2))
        intersect = where[0][int(len(where[0]) / 2)]
        ax.scatter(
            threshold[intersect],
            tpr[intersect],
            c="black",
            marker="*",
            s=100,
            label="Optimal Threshold",
            zorder=3,
        )

        plt.xlabel("Threshold")
        plt.title(f"Optimal Threshold: {title}")
        plt.xlim(0, 1)
        plt.legend()

        if save_name:
            plt.savefig(
                f"figures/cumulative_prediction/{save_name}_1.png",
                dpi=300,
            )
            plt.show()
        else:
            plt.show()

    print("Optimal thereshold is around:", threshold[intersect])

    if plot:
        fig, ax = plt.subplots(1, figsize=(7.5, 6))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.scatter(
            fpr,
            tpr,
            c=np.round(threshold, 3),
            s=6,
            cmap=sns.color_palette("Spectral", as_cmap=True),
        )  # )'cool')
        plt.clim(0, 1)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("Threshold", rotation=-90, va="bottom")

        plt.scatter(
            fpr[intersect],
            tpr[intersect],
            c="black",
            marker="*",
            s=100,
            label="Optimal Threshold",
        )

        plt.title(f"ROC Curve: {title}")
        plt.ylabel("TPR")
        plt.xlabel("FPR")
        plt.legend()

        if save_name:
            plt.savefig(
                f"figures/cumulative_prediction/{save_name}_2.png",
                dpi=300,
            )
        else:
            plt.show()

    return threshold[intersect]


# --------- Percentage of Predictions below Prediction against Predictions ---------


def plot_cumulative_predictions(
    predictions_dict, threshold, title=None, colors=None, save_name=None
):

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if colors == None:
        colors = [f"C{i}" for i in range(len(predictions_dict))]
    print(colors)

    i = 0
    for name, predictions in predictions_dict.items():
        preds = np.sort(predictions)
        sum_preds = [np.sum(preds < p) / len(preds) for p in preds]
        plt.plot(preds, sum_preds, linewidth=3, label=name, c=colors[i])
        i += 1

    if threshold:
        plt.plot(
            [threshold, threshold],
            [0, 1],
            linewidth=2,
            linestyle="dashed",
            c="grey",
            label=f"Optimal Threshold: {np.round(threshold,2)}",
        )

    plt.xlabel("Prediction, p")
    plt.ylabel(" % predictions < p")
    plt.title(f"{title}")
    plt.legend()

    if save_name:
        plt.savefig(
            f"figures/cumulative_prediction/{save_name}.png",
            dpi=300,
        )
    else:
        plt.show()


def plot_metrics(prediction_df):
    results = {"Random": {}, "Distributed": {}, "Structured": {}}
    # ROC curve
    q = int(len(prediction_df) / 4)
    ranges = {}
    ranges["Random"] = list(range(2 * q))
    r = list(range(q))
    r.extend(list(range(2 * q, 3 * q)))
    ranges["Distributed"] = r
    r = list(range(q))
    r.extend(list(range(3 * q, 4 * q)))
    ranges["Structured"] = r

    colors = {"Random": "C1", "Distributed": "C2", "Structured": "C3"}
    plt.figure()
    for sampling in ["Random", "Distributed", "Structured"]:
        y_true = prediction_df["True"][ranges[sampling]]
        y_pred = prediction_df["Best AUC Preds"][ranges[sampling]]
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
        results[sampling]["AUC"] = roc_auc_score(y_true, y_pred)
        # if sampling != 'Structured':
        plt.plot(
            fpr,
            tpr,
            color=colors[sampling],
            label=f"{sampling}, AUC={results[sampling]['AUC']:.4f}",
        )
        y_pred_neg = torch.tensor(y_pred[y_true == 0].tolist())
        y_pred_pos = torch.tensor(y_pred[y_true == 1].tolist())
        # HITS @ k
        for k in [20, 50, 100]:
            kth_score_in_negative_edges = torch.topk(y_pred_neg, k)[0][-1]
            results[sampling][f"HITS@{k}"] = float(
                torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()
            ) / len(y_pred_pos)

    display(pd.DataFrame(results))
    plt.plot([0, 1], [0, 1], color="black", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("Receiver Operating Characteristic")
    # plt.savefig(osp.join(path, f'roc.png'))
    plt.show()


def plot_prediction_distribution(prediction_df):
    results = {"Random": {}, "Distributed": {}, "Structured": {}}
    # ROC curve
    q = int(len(prediction_df) / 4)
    ranges = {}
    ranges["Random"] = list(range(2 * q))
    r = list(range(q))
    r.extend(list(range(2 * q, 3 * q)))
    ranges["Distributed"] = r
    r = list(range(q))
    r.extend(list(range(3 * q, 4 * q)))
    ranges["Structured"] = r

    colors = {"Random": "C1", "Distributed": "C2", "Structured": "C3"}

    # Distribution of predictions
    predictions_true = prediction_df[prediction_df["True"] == 1]["Best AUC Preds"]
    predictions_false = prediction_df[prediction_df["True"] == 0]["Best AUC Preds"]
    r = int(len(predictions_false) / 3)
    plt.figure()
    plt.hist(
        predictions_true,
        histtype="step",
        bins=50,
        alpha=0.6,
        color="C0",
        linewidth=2,
        label="Positive Class",
    )
    plt.hist(
        predictions_false[:r],
        histtype="step",
        alpha=0.6,
        bins=50,
        color=colors["Random"],
        linewidth=2,
        label="Random Negative Class",
    )
    plt.hist(
        predictions_false[r : 2 * r],
        histtype="step",
        alpha=0.6,
        bins=50,
        color=colors["Distributed"],
        linewidth=2,
        label="Distributed Negative Class",
    )
    plt.hist(
        predictions_false[2 * r :],
        histtype="step",
        alpha=0.6,
        bins=50,
        color=colors["Structured"],
        linewidth=2,
        label="Structured Negative Class",
    )
    plt.xlabel("Prediction")
    plt.ylabel("Count")
    plt.title("Predicted probability for each class.")
    plt.legend()
    # plt.savefig(osp.join(path, f'dist_prediction.png'))
    plt.show()

    results_df = pd.DataFrame.from_dict(results, orient="index")
    # results_df.to_csv(osp.join(path, f'test_metrics.csv'))
