# Tabular Regression with PyTorch Lightning

Assuming you have [PyTorch](https://pytorch.org/) and [Lightning](https://lightning.ai/docs/pytorch/latest/) installed, you can use this code to train a neural network regression on tabular data.

The goal of this repo is to give a starting point for ML practitioners which they can modify to suit their needs.

Features include:
1. Automatic creation of catogorical variables. Specify the `cat_cols` variable in [train_lightning_nn.py](./train_lightning_nn.py) and the code will handle the rest.
2. Has separate embedding layers for each categorical column.
3. A fairly rudimentary [3 layer neural network](./model.py) to start you off. **Note that you might need to scale your target variable to run this model.**

Enjoy!