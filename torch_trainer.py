import logging
import os
import os.path as osp
import time
import warnings
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from scipy.sparse import SparseEfficiencyWarning

import torch
from torch.nn import BCEWithLogitsLoss, Embedding
from torch_geometric.data import DataLoader
from torch_geometric.nn import GAE, VGAE
from torch_geometric.utils import to_undirected

from datasets.reaction_graph import ReactionGraph
from datasets.seal import SEALDynamicDataset
from datasets.GAE import GeneralDataset
from models.dgcnn import DGCNN
from models.autoencoder import (
    GCNEncoder,
    VariationalGCNEncoder,
    LinearEncoder,
    VariationalLinearEncoder,
)
from utils.metrics import mean_average_precision, hitsK

warnings.simplefilter("ignore", SparseEfficiencyWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphTrainer:
    """Class for handling training SEAL or GAE model for link prediction in reaction graph."""

    def __init__(self, settings):
        super(GraphTrainer, self).__init__()

        self.settings = settings
        self.n_runs = None
        self.run_it = 1
        self.running_best = {
            "Loss": {"Run": None, "Epoch": None, "Score": 100.0},
            "AUC": {"Run": None, "Epoch": None, "Score": 0.0},
            "last_epoch": {"Run": None, "Epoch": None},
        }
        # Evaluation metrics
        self.metrics = {
            "score": {"AUC": roc_auc_score, "AP": average_precision_score},
            # Metrics requiring a threshhold
            "prediction": {
                "Accuracy": accuracy_score,
                "Recall": recall_score,
                "Precision": precision_score,
                "F1": f1_score,
            },
        }
        # Place to store running and test scores
        self.scores = {
            "running": pd.DataFrame({}),  # run, epoch, score, metric, split
            "test": pd.DataFrame({}),
        }  # run, epoch, best on valid loss/auc score, metric
        self.predictions = pd.DataFrame({})
        # Store all related results
        trainer_id = time.strftime("%Y%m%d%H%M%S")
        self.res_dir = osp.join(f"results/{settings['name']}_{trainer_id}")

        if not osp.exists(self.res_dir):
            os.makedirs(self.res_dir)
        else:
            logging.warning("Warning: results will overwrite old files.")

        # Logging settings
        FORMAT = "%(asctime)s : %(levelname)s:%(module)s : %(funcName)s : %(message)s"
        DATEFMT = "%d/%b/%Y %H:%M:%S"
        logging.basicConfig(
            filename=osp.join(self.res_dir, "torch_trainer.log"),
            level=logging.DEBUG,
            format=FORMAT,
            datefmt=DATEFMT,
        )
        logging.info("Log file for trainer: %s_%s", settings["name"], trainer_id)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.axes._base").setLevel(logging.WARNING)
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)

        # set random seed for reproducibility
        torch.manual_seed(self.settings["seed"])
        np.random.seed(self.settings["seed"])

        # Save settings
        df_settings = pd.DataFrame.from_dict(
            self.settings, orient="index", columns=["Settings"]
        )
        df_settings.to_csv(osp.join(self.res_dir, "settings.csv"))

    def initialize_data(self, run_it):
        """Creates the an instance of ReactionGraph dataset class, considering given settings."""

        logging.info(
            "mode is: %s, valid %s, seed %s",
            self.settings["mode"],
            self.settings["valid_fold"],
            self.settings["seed"],
        )
        if (
            self.settings["mode"] == "cross_validation"
            or self.settings["mode"] == "cross_validation_5"
        ):
            if self.settings["splitting"] == "random":
                self.settings["valid_fold"] = run_it - 1
            else:
                self.settings["valid_fold"] = run_it
            # for cross-validation only positives will change within the split while negatives will not
            seed_neg_sampling = self.settings["seed"]

        elif self.settings["mode"] == "normal":
            # for run_it = 1 sampling will match the one when cross_valiation is done
            seed_neg_sampling = self.settings["seed"] + run_it - 1

        logging.info(
            "Initializing run %d. # %d fold is used as validation set. \
                     Neg sampling seed = %d.",
            run_it,
            self.settings["valid_fold"],
            seed_neg_sampling,
        )

        is_data = osp.isdir(f'data/{self.settings["name"]}')
        if is_data:
            os.system(f"rm -rf data/{self.settings['name']}")
            logging.info(
                "ReactionGraph is initialized. Saved in 'data/%s'.",
                self.settings["name"],
            )

        reaction_graph = ReactionGraph(
            f'data/{self.settings["name"]}', self.settings, seed_neg_sampling
        )
        self.settings["num_nodes"] = reaction_graph.num_nodes
        self.settings["features"] = reaction_graph.data.x
        reaction_graph.process_splits()

        return reaction_graph

    def make_data_splits(self, reaction_graph):
        """Creating the datasets and dataloaders for train, validation and test splits."""

        # Check for malicious settings
        if self.settings["valid_fold"] > 9 and self.settings["splitting"] == "random":
            logging.error(
                'Validation fold should be in range (0,9) when \
            splitting == "random".'
            )
        elif (
            self.settings["valid_fold"] > 9
            or self.settings["valid_fold"] < 1
            and self.settings["splitting"] == "time"
        ):
            logging.error(
                'Validation fold should be in range (1,9) when splitting == "time".'
            )

        if (
            self.settings["fraction_dist_neg"] > 1
            or self.settings["fraction_dist_neg"] < 0
        ):
            logging.error("'fraction_dist_neg' (%f) must be between 0 and 1.")

        # Train on train and validation data
        if "include_in_train" in self.settings.keys():
            include_in_train = self.settings["include_in_train"]

            train_pos_edge = (
                reaction_graph.data.split_edge["train"]["pos"].detach().clone()
            )
            train_neg_edge = (
                reaction_graph.data.split_edge["train"]["neg"].detach().clone()
            )
            valid_pos_edge = (
                reaction_graph.data.split_edge["valid"]["pos"].detach().clone()
            )
            valid_neg_edge = (
                reaction_graph.data.split_edge["valid"]["neg"].detach().clone()
            )
            test_pos_edge = (
                reaction_graph.data.split_edge["test"]["pos"].detach().clone()
            )
            test_neg_edge = (
                reaction_graph.data.split_edge["test"]["neg"].detach().clone()
            )
        else:
            self.settings["include_in_train"] = "train"
            include_in_train = self.settings["include_in_train"]

        if include_in_train == "valid":
            logging.info(
                "Training is done on train and valid set. \
            \n Test set is used for validation and testing."
            )
            # Concat train and valid and assign to train
            valid_pos_edge = to_undirected(valid_pos_edge)
            reaction_graph.data.split_edge["train"]["pos"] = torch.cat(
                (train_pos_edge, valid_pos_edge), dim=1
            )
            reaction_graph.data.split_edge["train"]["neg"] = torch.cat(
                (train_neg_edge, valid_neg_edge), dim=1
            )
            # Assign test to valid, test and valid is then the same
            reaction_graph.data.split_edge["valid"]["pos"] = test_pos_edge
            reaction_graph.data.split_edge["valid"]["neg"] = test_neg_edge
            logging.info("Adding validation edges to train, setting validation to test")

        # Train on train, validation and test data
        elif include_in_train == "test":
            logging.info(
                'Training is done on all edges. No validation set or test set is used. \
            \n Model is saved at epoch "n_epochs".'
            )
            # Concat train, valid and test and assign to train
            valid_pos_edge = to_undirected(valid_pos_edge)
            test_pos_edge = to_undirected(test_pos_edge)
            reaction_graph.data.split_edge["train"]["pos"] = torch.cat(
                (train_pos_edge, valid_pos_edge, test_pos_edge), dim=1
            )
            reaction_graph.data.split_edge["train"]["neg"] = torch.cat(
                (train_neg_edge, valid_neg_edge, test_neg_edge), dim=1
            )
            # Assign dummy valid set
            reaction_graph.data.split_edge["valid"]["pos"] = torch.tensor([[], []])
            reaction_graph.data.split_edge["valid"]["neg"] = torch.tensor([[], []])
            # Assign dummy test set
            reaction_graph.data.split_edge["test"]["pos"] = torch.tensor([[], []])
            reaction_graph.data.split_edge["test"]["neg"] = torch.tensor([[], []])
            logging.info(
                "Adding validation and test edges to train, removing validation and test"
            )

        # Split edges and create the dataloaders
        splits = ["train", "valid", "test"]
        datasets = {}
        dataloaders = {}

        for split in splits:
            if self.settings["model"] == "DGCNN":
                datasets[split] = SEALDynamicDataset(
                    root="data/SEAL",
                    dataset=reaction_graph,
                    settings=self.settings,
                    split=split,
                )

                dataloaders[split] = DataLoader(
                    datasets[split],
                    batch_size=self.settings["batch_size"],
                    shuffle=(split == "train"),
                    num_workers=self.settings["num_workers"],
                )
            else:
                datasets[split] = GeneralDataset(
                    root="data/general",
                    dataset=reaction_graph,
                    settings=self.settings,
                    split=split,
                )

            if split == "train":
                logging.info(
                    "%s dataset contains %d (%d x2 positive + %d negative) edges.",
                    split.upper(),
                    len(datasets[split]),
                    datasets[split].pos_edge.shape[1] // 2,
                    datasets[split].neg_edge.shape[1],
                )
            else:
                logging.info(
                    "%s dataset contains %d (%d + %d negative) edges.",
                    split.upper(),
                    len(datasets[split]),
                    datasets[split].pos_edge.shape[1],
                    datasets[split].neg_edge.shape[1],
                )

        return datasets, dataloaders

    def initialize_model(self, datasets):
        """Initializeing the DGCNN model, optimizer, learning rate scheduler
        and any embeddings.
        """
        # Node embeddings
        if self.settings["use_embedding"]:
            emb = Embedding(
                self.settings["num_nodes"], self.settings["hidden_channels"]
            ).to(DEVICE)
        else:
            emb = None

        # Initialize classifier model
        torch.manual_seed(self.settings["seed"])
        if self.settings["model"] == "DGCNN":
            model = DGCNN(
                hidden_channels=self.settings["hidden_channels"],
                num_layers=self.settings["num_layers"],
                max_z=self.settings["max_z"],
                k=self.settings["sortpool_k"],
                train_dataset=datasets["train"],
                dynamic_train=False,
                use_feature=self.settings["use_attribute"],
                node_embedding=emb,
                graph_norm=self.settings["graph_norm"],
                batch_norm=self.settings["batch_norm"],
                dropout=self.settings["dropout"],
                seed=self.settings["seed"],
            ).to(DEVICE)
        elif self.settings["model"] == "GAE":
            if self.settings["use_attribute"] == True:
                num_features = datasets["train"].num_features
            else:
                num_features = self.settings["num_nodes"]
                # num_features = 1
            if self.settings["variational"] == False:
                if self.settings["linear"] == False:
                    model = GAE(
                        GCNEncoder(
                            in_channels=num_features,
                            out_channels=self.settings["out_channels"],
                            seed=self.settings["seed"],
                            dropout=self.settings["dropout"],
                        )
                    ).to(DEVICE)
                else:
                    model = GAE(
                        LinearEncoder(
                            in_channels=num_features,
                            out_channels=self.settings["out_channels"],
                            seed=self.settings["seed"],
                            dropout=self.settings["dropout"],
                        )
                    ).to(DEVICE)
            else:
                if self.settings["linear"] == False:
                    model = VGAE(
                        VariationalGCNEncoder(
                            in_channels=num_features,
                            out_channels=self.settings["out_channels"],
                            seed=self.settings["seed"],
                            dropout=self.settings["dropout"],
                        )
                    ).to(DEVICE)
                else:
                    model = VGAE(
                        VariationalLinearEncoder(
                            in_channels=num_features,
                            out_channels=self.settings["out_channels"],
                            seed=self.settings["seed"],
                            dropout=self.settings["dropout"],
                        )
                    ).to(DEVICE)

            if self.settings["learning_rate"] < 0.01:
                logging.info(
                    "Learning rate %f smaller than suggested 0.01 for GAE",
                    self.settings["learning_rate"],
                )
        else:
            logging.error("Model %s not valid option.", self.settings["model"])

        parameters = list(model.parameters())
        if self.settings["use_embedding"]:
            torch.nn.init.xavier_uniform_(emb.weight)
            parameters += list(emb.parameters())
        optimizer = torch.optim.Adam(
            params=parameters, lr=self.settings["learning_rate"]
        )

        # If provided, load pretrained model
        if self.settings["pre_trained_model_path"] is not None:
            # Update in how models / optimizers are saved and loaded
            if osp.isfile(self.settings["pre_trained_model_path"]):
                # try: # after update model saved as dicts
                checkpoint = torch.load(self.settings["pre_trained_model_path"])
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # except: # before update saved separately
            elif osp.isfile(
                osp.join(
                    self.settings["pre_trained_model_path"],
                    "best_AUC_model_checkpoint.pth",
                )
            ):
                model.load_state_dict(
                    torch.load(
                        osp.join(
                            self.settings["pre_trained_model_path"],
                            "best_AUC_model_checkpoint.pth",
                        )
                    )
                )
                optimizer.load_state_dict(
                    torch.load(
                        osp.join(
                            self.settings["pre_trained_model_path"],
                            "best_AUC_optimizer_checkpoint.pth",
                        )
                    )
                )
            else:
                logging.error("Cannot load pre-trained_model. Check path in settings.")

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=self.settings["decay"]
        )

        if self.run_it == 1:
            total_params = sum(p.numel() for param in parameters for p in param)
            logging.info("Total number of parameters is %d", total_params)

        return model, optimizer, lr_scheduler, emb

    def update_scores(self, evaluation, split, epoch, run_it, best_on):
        """Gets scores: loss and evaluation metrics given an evaluation object
        and appends to dataframe where this informaion is stored.
        """

        scores_df = pd.DataFrame({})
        Score = namedtuple("Score", "metric score split run_it epoch")

        y_prob = torch.sigmoid(evaluation.y_prob)
        y_true = evaluation.y_true
        links = evaluation.links

        # Add loss
        score = Score("Loss", evaluation.loss, split, run_it, epoch)
        scores_df = add_score(scores_df, score)

        # Add metrics
        for metric, func in self.metrics["score"].items():
            score = Score(metric, func(y_true, y_prob), split, run_it, epoch)
            scores_df = add_score(scores_df, score)

        if self.settings["p_threshold"] == "roc":
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            max_roc_cutoff = thresholds[(tpr - fpr).argmax()]
            y_pred = (y_prob > max_roc_cutoff).int()
        else:
            y_pred = (y_prob > self.settings["p_threshold"]).int()
        for metric, func in self.metrics["prediction"].items():
            if metric == "Precision":
                score = Score(
                    metric, func(y_true, y_pred, zero_division=0), split, run_it, epoch
                )
            else:
                score = Score(metric, func(y_true, y_pred), split, run_it, epoch)

            scores_df = add_score(scores_df, score)

        TP = float(sum(y_pred[y_true == 1]) / sum(y_true == 1))
        FP = float(sum(y_pred[y_true == 0]) / sum(y_true == 0))
        TN = 1 - FP
        FN = 1 - TP
        rates = {"TPR": TP, "FPR": FP, "TNR": TN, "FNR": FN}
        for metric, rate in rates.items():
            score = Score(metric, rate, split, run_it, epoch)
            scores_df = add_score(scores_df, score)

        if split == "train" or split == "valid":
            self.scores["running"] = self.scores["running"].append(
                scores_df, ignore_index=True
            )
        else:
            # HITS @ k
            if len(y_pred) > 100:
                for k in [20, 50, 100]:
                    score = Score(
                        f"HITS@{k}", hitsK(y_true, y_pred, k), split, run_it, epoch
                    )
                    scores_df = add_score(scores_df, score)
            else:
                logging.warning("Too few datapoints for calculating HITS @ k")

            # MAP
            mean_ap, _, _ = mean_average_precision(y_true, y_pred, links)
            score = Score("MAP", mean_ap, split, run_it, epoch)
            scores_df = add_score(scores_df, score)

            scores_df = scores_df[["Run", "Epoch", "Metric", "Score", "Split"]]
            scores_df["Best on"] = best_on

            self.scores["test"] = self.scores["test"].append(
                scores_df, ignore_index=True
            )

    def train(self, model, dataloaders, datasets, loss_func, optimizer, emb):
        """Training function."""

        Evaluation = namedtuple("Evaluation", "loss auc y_true y_prob links")
        model.train()
        total_loss = 0
        y_score, y_true = [], []

        if self.settings["model"] == "DGCNN":
            for data in dataloaders["train"]:
                data = data.to(DEVICE)
                optimizer.zero_grad()
                logits = model(
                    data, use_feature=self.settings["use_attribute"], embedding=emb
                )
                y = data.y.to(torch.float)
                loss = loss_func(logits.view(-1).to(torch.float), y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * data.num_graphs
                y_score.append(logits.view(-1).detach().cpu())
                y_true.append(y.detach().cpu())

        elif self.settings["model"] == "GAE":
            optimizer.zero_grad()
            if self.settings["use_attribute"]:
                x = self.settings["features"].to(torch.float).to(DEVICE)
            else:
                x = torch.eye(self.settings["num_nodes"]).to(DEVICE)
                # x = torch.ones(self.settings['num_nodes'], 1).to(DEVICE) # should not be used because result depends on scalar value set
            pos_neg_edges = torch.cat(
                [datasets["train"].pos_edge, datasets["train"].neg_edge], 1
            ).to(DEVICE)
            pos_edges = datasets["train"].pos_edge.to(DEVICE)
            neg_edges = datasets["train"].neg_edge.to(torch.long).to(DEVICE)
            z = model.encode(x, datasets["train"].pos_edge.to(DEVICE))
            logits = model.decode(z, pos_neg_edges)
            loss = model.recon_loss(
                z, pos_edge_index=pos_edges, neg_edge_index=neg_edges
            )
            if self.settings["variational"] == True:
                loss = loss + (1 / self.settings["num_nodes"]) * model.kl_loss()

            y_true.append(torch.FloatTensor(datasets["train"].labels))

            loss.backward()
            optimizer.step()

            total_loss = loss.item()
            y_score.append(logits.view(-1).detach().cpu())

        y_true, y_score = torch.cat(y_true), torch.cat(y_score)

        total_loss /= len(y_true)
        auc = self.metrics["score"]["AUC"](y_true, y_score)

        return Evaluation(
            total_loss, auc, y_true.detach().clone(), y_score.detach().clone(), None
        )

    @torch.no_grad()
    def evaluate(self, model, split, dataloaders, datasets, loss_func, emb):
        """Evaluation function"""

        Evaluation = namedtuple("Evaluation", "loss auc y_true y_prob links")
        model.eval()
        total_loss = 0
        y_score, y_true = [], []
        links = []
        if self.settings["model"] == "DGCNN":
            for data in dataloaders[split]:
                data = data.to(DEVICE)
                logits = model(
                    data, use_feature=self.settings["use_attribute"], embedding=emb
                )
                loss = loss_func(logits.view(-1), data.y.to(torch.float))
                total_loss += loss.item() * data.num_graphs

                y_score.append(logits.view(-1).cpu())
                y_true.append(data.y.view(-1).cpu().to(torch.float))
                links.extend(data.link)
        elif self.settings["model"] == "GAE":
            if self.settings["use_attribute"]:
                x = self.settings["features"].to(torch.float).to(DEVICE)
            else:
                x = torch.eye(self.settings["num_nodes"]).to(DEVICE)
                # x = torch.ones(self.settings['num_nodes'], 1).to(DEVICE) # should not be used because result depends on scalar value set
            pos_neg_edges = torch.cat(
                [datasets[split].pos_edge, datasets[split].neg_edge], 1
            ).to(DEVICE)
            pos_edges = datasets[split].pos_edge.to(DEVICE)
            neg_edges = datasets[split].neg_edge.to(torch.long).to(DEVICE)
            z = model.encode(x, datasets["train"].pos_edge.to(DEVICE))
            logits = model.decode(z, pos_neg_edges)
            loss = model.recon_loss(
                z, pos_edge_index=pos_edges, neg_edge_index=neg_edges
            )
            if self.settings["variational"] == True:
                loss = loss + (1 / self.settings["num_nodes"]) * model.kl_loss()

            y_true.append(torch.FloatTensor(datasets[split].labels))

            total_loss = loss.item()
            y_score.append(logits.view(-1).cpu())
            links.extend(datasets[split].links)

        y_prob = torch.sigmoid(torch.cat(y_score))
        y_true = torch.cat(y_true)

        total_loss /= len(y_true)
        auc = self.metrics["score"]["AUC"](y_true, y_prob)

        return Evaluation(
            total_loss, auc, y_true.detach().clone(), y_prob.detach().clone(), links
        )

    @torch.no_grad()
    def predict(self, model, datasets, dataloader, split, emb):
        model.eval()
        y_score = []
        links = []
        if self.settings["model"] == "DGCNN":
            for data in dataloader[split]:
                data = data.to(DEVICE)
                logits = model(
                    data, use_feature=self.settings["use_attribute"], embedding=emb
                )
                y_score.append(logits.view(-1).cpu())
                links.extend(data.link)
        elif self.settings["model"] == "GAE":
            if self.settings["use_attribute"]:
                x = self.settings["features"].to(torch.float).to(DEVICE)
            else:
                x = torch.eye(self.settings["num_nodes"]).to(DEVICE)
            pos_neg_edges = torch.cat(
                [datasets[split].pos_edge, datasets[split].neg_edge], 1
            ).to(DEVICE)
            pos_edges = datasets[split].pos_edge.to(DEVICE)
            neg_edges = datasets[split].neg_edge.to(torch.long).to(DEVICE)
            z = model.encode(x, datasets["train"].pos_edge.to(DEVICE))
            logits = model.decode(z, pos_neg_edges)
            links.extend(datasets[split].links)
            y_score.append(logits.view(-1).cpu())

        y_prob = torch.sigmoid(torch.cat(y_score))

        return y_prob, links

    def run(self, running_test=True, final_test=False):
        """Full training process."""
        start = time.time()

        logging.info(
            "Starting training process on %s, results will be saved in %s.",
            DEVICE,
            self.res_dir,
        )

        # Settings depending on training mode
        assert self.settings["mode"] in [
            "normal",
            "cross_validation",
            "cross_validation_5",
        ], "'mode' setting invalid. Use: 'normal', 'cross_validation', 'cross_validation_5'"

        assert self.settings["splitting"] in [
            "time",
            "random",
        ], "'splitting' setting invalid. Chose from: 'time', 'random'"

        if self.settings["mode"] == "cross_validation":
            if self.settings["splitting"] == "random":
                self.n_runs = 10
                logging.debug("Random split and cross validation.")
            elif self.settings["splitting"] == "time":
                self.n_runs = 9
                logging.debug("Time split and cross validation.")
        elif self.settings["mode"] == "cross_validation_5":
            if self.settings["splitting"] == "random":
                self.n_runs = 5
                logging.debug("Random split and cross validation.")
            elif self.settings["splitting"] == "time":
                self.n_runs = 4
                logging.debug("Time split and cross validation.")
        elif self.settings["mode"] == "normal":
            self.n_runs = self.settings["n_runs"]
        else:
            logging.error("Training mode %s not a valid option.", self.settings["mode"])

        average_valid_auc = 0
        best_test_auc = 0
        runs_range = range(1, self.n_runs + 1)
        # Main training loop
        for run_it in tqdm(runs_range):
            self.run_it = run_it
            reaction_graph = self.initialize_data(self.run_it)
            datasets, dataloaders = self.make_data_splits(reaction_graph)
            model, optimizer, lr_scheduler, emb = self.initialize_model(datasets)
            loss_func = BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.settings["pos_weight_loss"])
            )

            if self.settings["model"] == "DGCNN":
                logging.debug(
                    "DGCNN use k=%f for a sortpool_k of %f.",
                    model.k,
                    self.settings["sortpool_k"],
                )

            best_in_run = {
                "Loss": {"Epoch": None, "Score": 100.0},
                "AUC": {"Epoch": None, "Score": 0.0},
            }

            for epoch in tqdm(range(1, self.settings["n_epochs"] + 1)):

                evaluation_train = self.train(
                    model, dataloaders, datasets, loss_func, optimizer, emb
                )
                self.update_scores(evaluation_train, "train", epoch, run_it, None)

                if (
                    len(datasets["valid"]) != 0
                ):  # True when training is done on all edges
                    evaluation_valid = self.evaluate(
                        model, "valid", dataloaders, datasets, loss_func, emb
                    )
                    self.update_scores(evaluation_valid, "valid", epoch, run_it, None)

                    logging.info(
                        "Epoch: %02d, Train Loss: %.4f, Valid Loss: %.4f, Valid AUC: %.4f",
                        epoch,
                        evaluation_train.loss,
                        evaluation_valid.loss,
                        evaluation_valid.auc,
                    )

                    if epoch == 1:
                        df_valid_set = pd.DataFrame(
                            {
                                "Source": [e[0] for e in evaluation_valid.links],
                                "Target": [e[1] for e in evaluation_valid.links],
                                "y true": evaluation_valid.y_true,
                            }
                        )
                        # Save validation set for each fold
                        if self.settings['mode'] == 'cross_validation_5':
                            df_valid_set.to_csv(
                                osp.join(self.res_dir, "validation_set_valid-fold=" + str(self.settings['valid_fold']) + ".csv")
                            )

                        elif run_it == 1:
                            df_valid_set.to_csv(
                                osp.join(self.res_dir, "validation_set.csv")
                            )

                    # Save checkpoints if validation performance has improved
                    if (
                        self.settings["include_in_train"] != "test"
                        and self.settings["include_in_train"] != "valid"
                    ):
                        for metric, score in (
                            ("Loss", evaluation_valid.loss),
                            ("AUC", evaluation_valid.auc),
                        ):

                            higher_score = score > best_in_run[metric]["Score"]
                            better = (
                                not higher_score if metric == "Loss" else higher_score
                            )

                            if (
                                better
                            ):  # and (self.settings['include_in_train'] != 'test' and self.settings['include_in_train'] != 'valid'):

                                best_in_run[metric] = {"Epoch": epoch, "Score": score}

                                save_torch_model(
                                    model,
                                    optimizer,
                                    emb,
                                    epoch,
                                    self.res_dir
                                    + f"/tmp_best_{metric}_model_checkpoint.pth",
                                )

                                if self.settings["mode"] == "cross_validation_5":
                                    save_torch_model(
                                        model,
                                        optimizer,
                                        emb,
                                        epoch,
                                        self.res_dir
                                        + f"/best_{metric}_model_checkpoint_iteration={str(run_it)}.pth",
                                    )

                                higher_score = (
                                    score > self.running_best[metric]["Score"]
                                )
                                global_better = (
                                    not higher_score
                                    if metric == "Loss"
                                    else higher_score
                                )

                                if global_better:
                                    self.running_best[metric] = {
                                        "Run": run_it,
                                        "Epoch": epoch,
                                        "Score": score,
                                    }
                                    save_torch_model(
                                        model,
                                        optimizer,
                                        emb,
                                        epoch,
                                        self.res_dir
                                        + f"/best_{metric}_model_checkpoint.pth",
                                    )

                # save model after last epoch for first run
                if run_it == 1:
                    save_torch_model(
                        model,
                        optimizer,
                        emb,
                        epoch,
                        self.res_dir + "/best_last_epoch_model_checkpoint.pth",
                    )
                lr_scheduler.step()
            average_valid_auc += best_in_run["AUC"]["Score"]

            if final_test and self.settings["include_in_train"] != "test":
                metric = "final_model"
                evaluation_final_test = self.evaluate(
                    model, "test", dataloaders, datasets, loss_func, emb
                )
                self.update_scores(
                    evaluation_final_test, "final epoch test", epoch, run_it, metric
                )
                df_preds = pd.DataFrame(
                    {
                        "Source": [e[0] for e in evaluation_final_test.links],
                        "Target": [e[1] for e in evaluation_final_test.links],
                        "y true": evaluation_final_test.y_true,
                        "y pred": evaluation_final_test.y_prob,
                        "run": run_it,
                        "model": "final epoch",
                        "based on": None,
                    }
                )
                self.predictions = self.predictions.append(df_preds)

            if running_test and (
                self.settings["include_in_train"] != "test"
                and self.settings["include_in_train"] != "valid"
            ):
                # Test current runs best models
                for metric in self.running_best:
                    metric_model_path = (
                        self.res_dir + f"/tmp_best_{metric}_model_checkpoint.pth"
                    )
                    if osp.isfile(metric_model_path):
                        model_checkpoint_path = metric_model_path
                        checkpoint = torch.load(model_checkpoint_path)
                        model.load_state_dict(checkpoint["model_state_dict"])
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        model_epoch = checkpoint["epoch"]
                        if self.settings["use_embedding"]:
                            emb.load_state_dict(checkpoint["embedding_state_dict"])

                        evaluation_test = self.evaluate(
                            model, "test", dataloaders, datasets, loss_func, emb
                        )
                        self.update_scores(
                            evaluation_test,
                            "best model test",
                            model_epoch,
                            run_it,
                            metric,
                        )

                        df_preds = pd.DataFrame(
                            {
                                "Source": [e[0] for e in evaluation_test.links],
                                "Target": [e[1] for e in evaluation_test.links],
                                "y true": evaluation_test.y_true,
                                "y pred": evaluation_test.y_prob,
                                "run": run_it,
                                "model": "highest metric",
                                "based on": [metric for y in evaluation_test.y_prob],
                            }
                        )
                        self.predictions = self.predictions.append(df_preds)

                for metric in best_in_run:
                    logging.info(
                        f"Best validation %s: %f, at epoch %d.",
                        metric,
                        best_in_run[metric]["Score"],
                        best_in_run[metric]["Epoch"],
                    )
            logging.info(
                "Finished run %d / %d of %d epochs.", run_it, self.n_runs, epoch
            )

        average_valid_auc /= len(runs_range)

        if (running_test or final_test) and len(datasets["test"]) != 0:
            self.scores["test"].to_csv(osp.join(self.res_dir, "test_scores.csv"))

        self.scores["running"].to_csv(osp.join(self.res_dir, "running_scores.csv"))
        self.predictions.to_csv(osp.join(self.res_dir, "test_predictions.csv"))
        plot_results(
            final_test=final_test, running_test=running_test, path=self.res_dir
        )

        logging.info("Finished all runs.")
        logging.info(
            "Overall best validation loss: %f", self.running_best["Loss"]["Score"]
        )
        logging.info(
            "Overall best validation AUC: %f", self.running_best["AUC"]["Score"]
        )

        for metric in self.running_best:
            tmp_checkpoint = osp.join(
                self.res_dir, f"tmp_best_{metric}_model_checkpoint.pth"
            )
            print(tmp_checkpoint)
            if osp.isfile(tmp_checkpoint):
                os.remove(tmp_checkpoint)

        end = time.time()
        m, s = divmod(end - start, 60)
        h, m = divmod(m, 60)
        logging.info(f"Took h:m:s %d:%d:%d", h, m, s)

        return average_valid_auc


# ------------------------- Functions -------------------------


def add_score(scores_df, score):
    """Helper function for updating scores."""

    tmp = pd.DataFrame({})
    tmp["Run"] = [score.run_it]
    tmp["Epoch"] = [score.epoch]
    tmp["Score"] = [score.score]
    tmp["Metric"] = [score.metric]
    tmp["Split"] = [score.split]

    if scores_df is None:
        scores_df = pd.DataFrame({})

    return scores_df.append(tmp, ignore_index=True)


def save_torch_model(model, optimizer, emb, epoch, save_as):
    """Helper function for saving model, optimizer and if provided embeddings."""

    if emb:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "embedding_state_dict": emb.stat_dict(),
                "epoch": epoch,
            },
            save_as,
        )
    else:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            },
            save_as,
        )


def plot_calibration_curve(y_true, y_pred, label="Model"):
    """Plot the calibration curve for the model."""

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    fraction_positives, mean_predicted_value = calibration_curve(
        y_true=y_true, y_prob=y_pred, n_bins=20
    )
    ax1.plot(mean_predicted_value, fraction_positives, "s-", label=label)

    ax2.hist(y_pred, range=(0, 1), bins=100, label=label, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title("Calibration plots  (reliability curve)")

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()

    return fig, fraction_positives, mean_predicted_value


def plot_results(final_test, running_test, path=None):
    """Plot the results in various ways.
    Args:
        final_test (boolean): If True plot result on test set.
        running_test (boolean): Was running_test used during runs.
        path (string): Pathway to result directory.
    """

    scores = pd.read_csv(osp.join(path, "running_scores.csv"))
    # Plot Loss
    loss_scores = scores[scores["Metric"] == "Loss"]
    fig = sns.relplot(data=loss_scores, x="Epoch", y="Score", hue="Split", kind="line")
    fig.set(ylabel="Loss")
    fig.savefig(osp.join(path, "loss.png"))
    plt.close()

    # Plot Metrics
    columns = ["AUC", "AP", "Accuracy", "F1", "Recall", "Precision"]
    fig = sns.relplot(
        data=scores,
        x="Epoch",
        y="Score",
        hue="Split",
        kind="line",
        col="Metric",
        col_wrap=3,
        col_order=columns,
    )
    fig.savefig(osp.join(path, "metrics.png"))
    plt.close()

    if final_test:
        # ROC curve
        test_df = pd.read_csv(osp.join(path, "test_predictions.csv"), index_col=0)

        plt.figure()

        if running_test and (
            self.settings["include_in_train"] != "test"
            and self.settings["include_in_train"] != "valid"
        ):
            test_df = test_df[test_df["model"] == "highest metric"]
            test_df = test_df[test_df["based on"] == "AUC"]
        y_true = test_df["y true"]
        y_pred = test_df["y pred"]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.plot([0, 1], [0, 1], color="black", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.title("Receiver Operating Characteristic")
        plt.savefig(osp.join(path, "roc.png"))
        plt.close()

        # Distribution of predictions
        predictions_true = test_df[test_df["y true"] == 1]["y pred"]
        predictions_false = test_df[test_df["y true"] == 0]["y pred"]

        plt.figure()
        plt.hist(
            predictions_true,
            histtype="step",
            bins=50,
            alpha=0.6,
            color="green",
            linewidth=2,
            label="Positive Class",
        )
        plt.hist(
            predictions_false,
            histtype="step",
            alpha=0.6,
            bins=50,
            color="red",
            linewidth=2,
            label="Negative Class",
        )

        plt.xlabel("Prediction")
        plt.ylabel("Count")
        plt.title("Predicted probability for each class.")
        plt.legend()
        plt.savefig(osp.join(path, "dist_prediction.png"))
        plt.close()

        # Make calibration curve plot
        fig, _, _ = plot_calibration_curve(y_true, y_pred, label="SEAL Model")
        fig.savefig(osp.join(path, "calibration_curve.png"))
