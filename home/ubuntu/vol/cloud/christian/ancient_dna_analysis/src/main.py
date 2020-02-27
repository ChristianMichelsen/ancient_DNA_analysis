import os
import pysam
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import pandas as pd
from timer import Timer
from importlib import reload
import yaml

import extra_funcs
import bam_reader
from data import dataframe_loader
from viz import visualize


import wandb


# https://colab.research.google.com/github/wandb/examples/blob/master/pytorch-intro/intro.ipynb#scrollTo=bZpt5W2NNl6S
wandb.init(project="ancient_dna_analysis")

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config  # Initialize config
config.batch_size = 4  # input batch size for training (default: 64)
config.test_batch_size = 10  # input batch size for testing (default: 1000)
config.epochs = 50  # number of epochs to train (default: 10)
config.lr = 0.1  # learning rate (default: 0.01)
config.momentum = 0.1  # SGD momentum (default: 0.5)
config.no_cuda = False  # disables CUDA training
config.seed = 42  # random seed (default: 42)
config.log_interval = 10  # how many batches to wait before logging training status


# from viz.visualize import viz_test
np.random.seed(42)

paths, cfg = extra_funcs.load_yamls()

N_chromosones = cfg["N_chromosones"]
num_cores = cfg["num_cores"]


chromosones = [f"chr{i}" for i in range(1, N_chromosones + 1)]

df_seq = dataframe_loader.load_seqs(chromosones, num_cores)

reload(extra_funcs)
X, y = extra_funcs.get_X_and_y(df_seq, chromosones)


train_frac = 0.8
train_idx = int(len(X) * train_frac)
test_idx = len(X) - train_idx

from torch.utils.data import TensorDataset, DataLoader

# split dataset into train and valid TensorDataset for NN training
train_ds = TensorDataset(X[:train_idx], y[:train_idx])  # FIRST 1500 data points
valid_ds = TensorDataset(X[-test_idx:], y[-test_idx:])


# verify train and valid dataset length
len(train_ds), len(valid_ds)


from torch import nn

# set dropout
drop_p = 0.2

# set batch size
bs = 64

N_epochs = 100

net_basic = nn.Sequential(
    nn.Conv1d(in_channels=4, out_channels=32, kernel_size=12),
    nn.MaxPool1d(kernel_size=4),
    nn.Flatten(),
    nn.Dropout(drop_p),
    nn.Linear(in_features=512, out_features=16),
    nn.ReLU(),
    nn.Dropout(drop_p),
    nn.Linear(in_features=16, out_features=2),
    # Debugger() # optional debugger layer
)

if cfg["use_gpu"]:
    net_basic = net_basic.cuda()

# check network architecture
net_basic

for p in net_basic.parameters():
    print(p.device)


# Magic
wandb.watch(net_basic, log="all")


# setup DataLoader for NN training from TensorDatasets for NN training
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False)


# set optimizer type, parameters, and hyperparameters
opt = torch.optim.SGD(net_basic.parameters(), lr=1e-3, momentum=0.9)


# import torch.functional as F
import torch.nn.functional as F


filename_trained_model = paths["trained_model"].format(
    N_chromosones=N_chromosones, N_epochs=N_epochs
)


if cfg["force_rerun"] or not Path(filename_trained_model).is_file():

    reload(extra_funcs)

    with Timer("Fit:"):
        (
            epoch_list,
            train_loss_list,
            valid_loss_list,
            acc_list,
            y_hat_list,
            y_true_list,
        ) = extra_funcs.simple_fit(
            N_epochs, net_basic, F.binary_cross_entropy_with_logits, opt, train_dl, valid_dl
        )

    torch.save(net_basic.state_dict(), filename_trained_model)
    wandb.save("model.h5")

    plt.subplot(2, 1, 1)
    plt.title("Training")
    plt.ylabel("Loss")
    plt.plot(train_loss_list, label="Train loss")
    plt.plot(valid_loss_list, label="Valid loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel("Training batch")
    plt.ylabel("Accuracy")
    plt.plot(acc_list, label="Accuracy metric")
    plt.legend()

    plt.show()

    plt.savefig("testfig.pdf")

else:
    net_basic.load_state_dict(torch.load(filename_trained_model))


net_basic.eval()


with torch.no_grad():
    # calculations for accuracy_thres metric:
    y_hat = net_basic(X[-test_idx:]).cpu().numpy()
    y_true = y[-test_idx:].cpu().numpy()


import itertools

from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(y_true[:, 0], (y_hat > 0).astype("int")[:, 0])

# From https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

# plot_confusion_matrix(cnf_matrix, classes=[0, 1])
reload(visualize)
visualize.plot_confusion_matrix(cnf_matrix, classes=[0, 1], normalize=True)

