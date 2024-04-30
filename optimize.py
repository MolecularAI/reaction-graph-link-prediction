
import os
import sys
import json
import argparse
import importlib

import optuna
from optuna.samplers import TPESampler
from torch_trainer import GraphTrainer

sys.path.insert(0, "settings")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Parse argments for updating
parser = argparse.ArgumentParser(description="ELN SEAL")
parser.add_argument("-n", "--name", type=str, default=None)
parser.add_argument("-g", "--graph_path", type=str, default=None)
parser.add_argument("-t", "--n_trials", type=int, default=10)
parser.add_argument("-a", "--study_name", type=str, default='HyperparameterTuning')
parser.add_argument("-p", "--optimisation_parameters_path", type=str, default='settings/optuna.py')
# Subgraphs in SEAL
parser.add_argument("--num_hops", type=int, default=1)
parser.add_argument("--ratio_per_hop", type=float, default=1.0)
parser.add_argument("--max_nodes_per_hop", type=int, default=800)
parser.add_argument("--node_label", type=str, default='drnl')
# Datasplits
parser.add_argument("--neg_pos_ratio_test", type=float, default=1.0)
parser.add_argument("--neg_pos_ratio", type=float, default=1.0)
parser.add_argument("--train_fraction", type=float, default=1.0)
parser.add_argument("--splitting", type=str, default='random')
parser.add_argument("--valid_fold", type=int, default=1)
parser.add_argument("--fraction_dist_neg", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--include_in_train", type=str, default=None)
parser.add_argument("--mode", type=str, default='normal')
# Dataloaders of size and number of threads
parser.add_argument("-bs", "--batch_size", type=int, default=256)
parser.add_argument("--num_workers", type=int, default=6)
# NN hyperparameters
parser.add_argument("--model", type=str, default='DGCNN')
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.0005)
parser.add_argument("--decay", type=float, default=0.855)
parser.add_argument("--dropout", type=float, default=0.517)
parser.add_argument("--n_runs", type=int, default=1)
# SEAL training hyperparameters
parser.add_argument("--hidden_channels", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=6)
parser.add_argument("--max_z", type=int, default=1000)
parser.add_argument("--sortpool_k", type=float, default=879)
parser.add_argument("--graph_norm", action="store_true")
parser.add_argument("--batch_norm", action="store_true")
# GAE training hyperparameters
parser.add_argument("--variational", action="store_true")
parser.add_argument("--linear", action="store_true")
parser.add_argument("--out_channels", type=int, default=None)
# Graph options
parser.add_argument("--use_attribute", type=str, default='fingerprint')
parser.add_argument("--use_embedding", action="store_true")
# Classification parameters
parser.add_argument("--p_threshold", type=float, default=0.9)
parser.add_argument("--pos_weight_loss", type=float, default=1.0)

args = parser.parse_args()
settings = vars(args)

n_trials = args.n_trials
study_name = args.study_name

if not os.path.exists(args.optimisation_parameters_path):
    params_dictionary = {}
    if settings["model"] == "DGCNN":
        params_dictionary["num_hops"] = [1, 2]
        params_dictionary["sortpool_k"] = [1, 1000]
        params_dictionary["hidden_channels"] = [4, 8, 16, 32, 64, 128, 256]
        params_dictionary["num_layers"] = [1, 12]
    elif settings["model"] == "GAE":
        params_dictionary["out_channels"] = [4, 8, 16, 32, 64, 128, 256]
        params_dictionary["linear"] = [True, False]
        params_dictionary["variational"] = [True, False]
    else:
        print("Error: Unknown model name.")
        sys.exit()

    # common parameters
    params_dictionary["learning_rate"] = [0.0001, 0.001, 0.01, 0.1]
    params_dictionary["dropout"] = [0.0, 0.99]
    params_dictionary["decay"] = [0.0, 0.99]
    params_dictionary["startup_trials"] = 50

    json.dump(params_dictionary, open(args.optimisation_parameters_path, "w"))
    #print("Done writing optimisation parameters. Exiting.")
    #sys.exit()
else:
    params_dictionary = json.load(open(args.optimisation_parameters_path))

# Determine path of graph based on above settings
assert settings["graph_path"] is not None, "-g --graph_path not provided as input or in settings file"


def objective(trial):

    # setting parameters and the ranges for tuning
    if settings["model"] == "DGCNN":
        num_hops = trial.suggest_int(
            "num_hops",
            params_dictionary["num_hops"][0],
            params_dictionary["num_hops"][1],
        )
        sortpool_k = trial.suggest_int(
            "sortpool_k",
            params_dictionary["sortpool_k"][0],
            params_dictionary["sortpool_k"][1],
        )
        hidden_channels = trial.suggest_categorical(
            "hidden_channels", params_dictionary["hidden_channels"]
        )
        num_layers = trial.suggest_int(
            "num_layers",
            params_dictionary["num_layers"][0],
            params_dictionary["num_layers"][1],
        )

        settings["num_hops"] = num_hops
        settings["sortpool_k"] = sortpool_k
        settings["hidden_channels"] = hidden_channels
        settings["num_layers"] = num_layers
    elif settings["model"] == "GAE":
        out_channels = trial.suggest_categorical(
            "out_channels", params_dictionary["out_channels"]
        )
        linear = trial.suggest_categorical("linear", params_dictionary["linear"])
        variational = trial.suggest_categorical(
            "variational", params_dictionary["variational"]
        )

        settings["out_channels"] = out_channels
        settings["linear"] = linear
        settings["variational"] = variational
    else:
        print("Error: Unknown model name.")
        sys.exit()

    # common parameters
    learning_rate = trial.suggest_categorical(
        "learning_rate", params_dictionary["learning_rate"]
    )
    dropout = trial.suggest_float(
        "dropout", params_dictionary["dropout"][0], params_dictionary["dropout"][1]
    )
    decay = trial.suggest_float(
        "decay", params_dictionary["decay"][0], params_dictionary["decay"][1]
    )
    settings["learning_rate"] = learning_rate
    settings["dropout"] = dropout
    settings["decay"] = decay
    settings["name"] = f"{study_name}/trial_{trial.number}"

    trainer = GraphTrainer(settings)
    average_valid_auc = trainer.run(running_test=False, final_test=False)

    return average_valid_auc


# Create the OPTUNA study
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
    direction="maximize",
    sampler=TPESampler(n_startup_trials=params_dictionary["startup_trials"]),
)

study.optimize(objective, n_trials=n_trials)

