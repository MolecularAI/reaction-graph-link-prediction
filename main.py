
import os
import argparse
import importlib

from torch_trainer import GraphTrainer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Parse argments for updating
parser = argparse.ArgumentParser(description="SEAL")
parser.add_argument("-s", "--settings_path", type=str, default=None)
parser.add_argument("-n", "--name", type=str, default="no_name")
parser.add_argument("-g", "--graph_path", type=str, default=None)
parser.add_argument("--pre_trained_model_path", type=str, default=None)
# Subgraphs in SEAL
parser.add_argument("--num_hops", type=int, default=1)
parser.add_argument("--ratio_per_hop", type=float, default=1.0)
parser.add_argument("--max_nodes_per_hop", type=int, default=800)
parser.add_argument("--node_label", type=str, default='drnl')
# Datasplits
parser.add_argument("--neg_pos_ratio_test", type=float, default=1.0)
parser.add_argument("--neg_pos_ratio", type=float, default=1.0)
parser.add_argument("--train_fraction", type=float, default=0.8)
parser.add_argument("--test_fraction", type=float, default=0.2)
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

# Read settings from supplied settings.txt file
if args.settings_path is not None:
    module_path = str(args.settings_path)
    settings_module = importlib.import_module(module_path)
    settings = settings_module.settings
else:
    settings = {arg: value for arg, value in vars(args).items()}

if settings["model"] == "DGCNN":
    if "max_nodes_per_hop" not in settings:
        settings["max_nodes_per_hop"] = None

# Determine path of graph based on above settings
if args.graph_path is not None:
    settings["graph_path"] = args.graph_path
elif "graph_path" not in settings:
    raise ValueError("-g --graph_path not provided as input or in settings file")

trainer = GraphTrainer(settings)
_ = trainer.run(running_test=False, final_test=True)

