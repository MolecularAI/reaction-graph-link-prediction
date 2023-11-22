# Reaction Graph Link Prediction

This repository contains end-to-end training and evaluation of the SEAL [[1]](https://proceedings.neurips.cc/paper_files/paper/2018/file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf) and Graph Auto-Encoder [[2]](https://arxiv.org/abs/1611.07308) link prediction algorithms on a Chemical Reaction Knowledge Graph built on reactions from USPTO. This code has been used to generate the results in [[3]](https://chemrxiv.org/engage/chemrxiv/article-details/64e34fe400bbebf0e68bcfb8).

## Reaction Prediction 

In [3], a novel de novo design method is presented in which the link prediction is used for predicting novel pairs of reactants. The link prediction is then followed by product prediction using a transformer model, Chemformer, which predicts the products given the reactants. This repository covers the link prediction (reaction prediction) and for the subsequent product prediction we refer to the original [Chemformer](https://github.com/MolecularAI/Chemformer) repository. 

Link Prediction in this setting is equivalent to predicting novel reactions between reactant pairs. The code presented here is based on Muhan Zhang's [implementation of SEAL](https://github.com/facebookresearch/SEAL_OGB/tree/main).


## Running the code

To train the model, run ```main.py```. Settings for the run, such as graph path, hyperparameters and paths are provided as parameter file and/or parameters.
```optimize.py``` is used for the hyperparameter optimization. 
```predict_links.py``` is used for predicting the probability of links given a trained model.

When run ```main.py``` generates:
- data: Processed data files.
- results: Individual folders containing all relevant results from a GraphTrainer, including
    - Checkpoints of model and optimizer parameters, based on best validation AUC and best validation loss separately. 
    - Log file of outputs from training, including number of datapoints in train/valid/test split, number of network parameters and more. 
    - Pickle files of all results from training and testing separately. 
    - Some preliminary plots.
    - Test metrics and test predictions in csv format. 
    - A csv settings file of the hyperparameters used for training.

## Codebase

```torch_trainer.py``` contains the main trainer class and is called by the ```main.py```, ```optimize.py``` and ```predict_links.py``` individually.

The main script initializes and runs a GraphTrainer from the ```torch_trainer.py``` file. The training process utilizes the following modules:
- ```datasets/reaction_graph.py```: Importing graph and setting up training/validation/test positive edges. 
- ```datasets/seal.py```: Dynamic dataloader for SEAL algorithm, including sub-graph extraction and Double Radius Node Labelling (DRNL).
- ```datasets/GAE.py```: Dataloader for GAE algorithm.
- ```models/dgcnn:``` The Deep Graph Convolutional Neural Networks used for prediction of the likelihood of a link between the source and target nodes in the given subgraph.
- ```models/autoencoder:``` Graph Autoencoder used for prediction of the likelihood of a link between the source and target nodes, implemented using Torch Geometric library.
- ```utils```: various related functions used throughout the project. 


## Requirements
A list of dependancies can be found in **conda_env_list/link_prediction.yaml**.

## Parallelization / Runtime
Best run with GPU available, in addition SEAL-based link prediction is paralelizeable on CPUs. Negative links generation by default uses a node degree distribution-preserving sampling function (sample_degree_preserving_distribution) which can take long time depending on graph size, however it only needs to be run once for a given link-sampling seed after which it is stored in data folder; alternatively an approximating function (sample_distribution) can be used with quicker runtime.

## Data
The reaction graph used in [1] is available here: <https://doi.org/10.5281/zenodo.10171188>.

## Contributors
- Emma Rydholm
- Tomas Bastys
- Emma Svensson


## References:
[1] M. Zhang and Y. Chen, "Link prediction based on graph neural networks," Advances in neural information processing systems 31, 2018.

[2] T. N. Kipf and M. Welling, "Variational Graph Auto-Encoders", Neural Information Processing Systems 2016.

[3] E. Rydholm, T. Bastys, E. Svensson, C. Kannas, O. Engkvist and T. Kogej, "Expanding the chemical space using a Chemical Reaction Knowledge Graph,"  ChemRxiv. 2023

[4] R. Irwin, S. Dimitriadis, J. He and E. Bjerrum, "Chemformer: a pre-trained transformer for computational chemistry," Machine Learning: Science and Technology. 2022, 31 Jan. 2022. 
