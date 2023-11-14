import os
import argparse
import importlib

import sys

sys.path.insert(0, "settings")

from settings import settings

from torch_trainer import GraphTrainer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Parse argments for updating
parser = argparse.ArgumentParser(description="SEAL")
parser.add_argument("-s", "--settings_path", type=str, default=None)
parser.add_argument("-n", "--name", type=str, default=None)
parser.add_argument("-g", "--graph_path", type=str, default=None)
# Subgraphs in SEAL
parser.add_argument("--num_hops", type=int, default=None)
parser.add_argument("--ratio_per_hop", type=float, default=None)
parser.add_argument("--max_nodes_per_hop", type=int, default=None)
parser.add_argument("--node_label", type=str, default=None)
# Datasplits
parser.add_argument("--neg_pos_ratio_test", type=float, default=None)
parser.add_argument("--neg_pos_ratio", type=float, default=None)
parser.add_argument("--train_fraction", type=float, default=None)
parser.add_argument("--test_fraction", type=float, default=None)
parser.add_argument("--splitting", type=str, default=None)
parser.add_argument("--valid_fold", type=int, default=None)
parser.add_argument("--fraction_dist_neg", type=float, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--include_in_train", type=str, default=None)
parser.add_argument("--mode", type=str, default=None)
# Dataloaders of size and number of threads
parser.add_argument("-bs", "--batch_size", type=int, default=None)
parser.add_argument("--num_workers", type=int, default=None)
# NN hyperparameters
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--n_epochs", type=int, default=None)
parser.add_argument("-lr", "--learning_rate", type=float, default=None)
parser.add_argument("--decay", type=float, default=None)
parser.add_argument("--dropout", type=float, default=None)
parser.add_argument("--n_runs", type=int, default=None)
# SEAL training hyperparameters
parser.add_argument("--hidden_channels", type=int, default=None)
parser.add_argument("--num_layers", type=int, default=None)
parser.add_argument("--max_z", type=int, default=None)
parser.add_argument("--sortpool_k", type=float, default=None)
parser.add_argument("--graph_norm", action="store_true")
parser.add_argument("--batch_norm", action="store_true")
# GAE training hyperparameters
parser.add_argument("--variational", action="store_true")
parser.add_argument("--linear", action="store_true")
parser.add_argument("--out_channels", type=int, default=None)
# Graph options
parser.add_argument("--use_attribute", action="store_true")
parser.add_argument("--use_embedding", action="store_true")
# Classification parameters
parser.add_argument("--p_threshold", type=float, default=None)

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
    if value is not None and value is not False:
        settings[arg] = value

if settings["model"] == "DGCNN":
    if "max_nodes_per_hop" not in settings:
        settings["max_nodes_per_hop"] = None

# Set name of directory to store logs and results
if args.name is not None:
    settings["name"] = args.name
else:
    settings["name"] = "no_name"

# Determine path of graph based on above settings
if args.graph_path is not None:
    settings["graph_path"] = args.graph_path
elif "graph_path" not in settings:
    raise ValueError("-g --graph_path not provided as input or in settings file")

trainer = GraphTrainer(settings)
_ = trainer.run(running_test=False, final_test=True)
