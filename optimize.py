import os
import argparse
import importlib

import optuna
from optuna.samplers import TPESampler, RandomSampler
from torch_trainer import GraphTrainer

import sys

sys.path.insert(0, "settings")
import json

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Parse argments for updating
parser = argparse.ArgumentParser(description="ELN SEAL")
parser.add_argument("-s", "--settings_path", type=str, default=None)
parser.add_argument("-n", "--name", type=str, default=None)
parser.add_argument("-g", "--graph_path", type=str, default=None)
parser.add_argument("-t", "--n_trials", type=int, default=10)
parser.add_argument("-a", "--study_name", type=str, default="HyperparameterTuning")
parser.add_argument("-p", "--optimisation_parameters_path", type=str, default=None)

args = parser.parse_args()

# Read settings from supplied settings.txt file
if args.settings_path is not None:
    module_path = str(args.settings_path)
    settings_module = importlib.import_module(module_path)
    settings = settings_module.settings
else:
    settings = {}

# Update settings based on remaining arguments
for arg, value in vars(args).items():
    if arg == "settings_path":
        continue
    elif value is not None and value is not False:
        settings[arg] = value

if settings["model"] == "DGCNN":
    if "max_nodes_per_hop" not in settings:
        settings["max_nodes_per_hop"] = None

n_trials = args.n_trials
study_name = args.study_name

# Parameter file for optimisation
if args.optimisation_parameters_path is None:
    raise ValueError("-p --optimisation_parameters_path not provided as input")

write_params_to_file = False

if write_params_to_file:
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
    print("Done writing optimisation parameters. Exiting.")
    sys.exit()
else:
    params_dictionary = json.load(open(args.optimisation_parameters_path))

# Determine path of graph based on above settings
if args.graph_path is not None:
    settings["graph_path"] = args.graph_path
elif "graph_path" not in settings:
    raise ValueError("-g --graph_path not provided as input or in settings file")


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
