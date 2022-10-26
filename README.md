# Geodesic Graph Neural Network for Efficient Graph Representation Learning

This is the official codebase of the paper

[Geodesic Graph Neural Network for Efficient Graph Representation Learning](https://arxiv.org/abs/2210.02636)

Lecheng Kong, Yixin Chen, Muhan Zhang

The package is developed based on the GNN toolbox gnnfree.

## Installation

We recommend installation from Conda:

```bash
git clone https://github.com/woodcutter1998/gdgnn.git
cd gdgnn
sh setup.sh
```

## Usage

`--gd_type` controls the type of geodesics, it can either be `VerGD` for vertical geodesics or `HorGD` for horizontal geodesics.

`--num_layers` controls the number of layers in the GNN.

To run different datasets, do `python run_**.py` with the parameters specified.

To search for hyperparameters, modify the `hparams` variable in the `run_**.py` files to specify the list of potentail hyperparameters. e.g.
```python
hparams = {'num_layers':{2,3,4,5},
            'gd_type':{'VerGD, HorGD'},
            'dropout':{0.5, 0.7, 0.9}}
``` 
and do `python run_**.py --psearch True`. The program performs grid-search on the hyperparameters specified.

An example command to reproduce our results on OGBG-MOLHIV dataset is:

```bash
python run_MOL.py --train_data_set ogbg-molhiv --psearch True
```