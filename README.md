# ML4G Project - Reproducibility Challenge
This repo contains code for running our experiments where we try to verify results from [Beyond Low-frequency
Information in Graph Convolutional Networks](https://arxiv.org/abs/2101.00797). You can find graphs for all our 
experiments [here](https://wandb.ai/yolomeus/ml4g-project/reports/ML4G-Project--VmlldzoxNDU5ODQz). The report for 
this project is located [here](report.pdf).

## Requirements
The code was run using python 3.9 in a virtual anaconda environment. All required packages can be found
in `environment.yml` and installed using [anaconda](anaconda.com/products/individual)
like so:

```shell
conda env create -f environment.yml
```

The packages listed can also be installed manually using pip. (Note: cudatoolkit is optional and only needed for gpu
training. Same goes for wandb if you do not wish to use wandb for logging)

## Datasets
The raw and pre-processed datasets can both be downloaded from 
[here](https://github.com/yolomeus/ml4g-project/releases/tag/data).


## Experiment Configuration

This project uses [hydra](https://github.com/facebookresearch/hydra) for composing a training configuration from
multiple sub-configuration modules. All configuration files can be found in `conf/` and its subdirectories, where
`config.yaml` is the main configuration file and subdirectories contain possible sub-configurations.

Even though yaml files define the default configuration and its structure, all training parameters can be overriden via
command line arguments when running the `train.py` script and sub-configurations can be replaced allowing easy plug and
play of training components.

### Examples

Replace the `datamodule`'s default `dataset` configuration with the `pubmed` configuration -> use `pubmed` for training:

```shell
python train.py datamodule/dataset=pubmed
```

Sweep over random seeds (multirun):

```shell
python -m train.py random_seed=4502928,884526,2926849
```

Do a grid search over learning rate and dataset combinations:

```shell
python -m train.py loop.optimizer.lr=0.01,0.1 datamodule/dataset=cora,pubmed,squirrel
```
 

