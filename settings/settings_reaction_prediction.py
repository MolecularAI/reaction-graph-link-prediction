settings = {
    # Subgraphs in SEAL
    'num_hops': 1,
    'ratio_per_hop': 1.0,
    'max_nodes_per_hop': 800,#None,
    'node_label': 'drnl',
    # Datasplits
    'seed': 100,
    'mode': 'normal',#'increase_negatives','fixed',
    'train_fraction': 1,
    'splitting': 'random',
    'valid_fold': 1,
    'neg_pos_ratio': 1, # How many percent to sample from distribution
    'neg_pos_ratio_test': 1,
    'fraction_dist_neg': 1, # How many percent to sample from distribution
    # Dataloaders of size and number of threads 
    'batch_size': 256,
    'num_workers': 6,
    # GNN hyperparameters
    'model': 'DGCNN',
    'hidden_channels': 128,
    'num_layers': 6,
    'max_z': 1000,
    'sortpool_k': 879, 
    'graph_norm': False,
    'batch_norm': False,
    'dropout': 0.517,
    'pre_trained_model_path': None,
    # SEAL training hyperparameters
    'n_epochs': 20,
    'learning_rate': 0.0005,
    'decay': 0.855,
    'n_runs': 1,
    ##### Graph options #####
    'use_attribute': 'fingerprint',
    'use_embedding': False,
    ##### Evaluation #####
    'p_threshold': 0.9,
    'pos_weight_loss': 1,
}
