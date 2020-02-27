from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import wandb


def load_yamls():
    with open("paths.yaml", "r") as file:
        paths = yaml.safe_load(file)
    with open("cfg.yaml", "r") as file:
        cfg = yaml.safe_load(file)
    return paths, cfg


paths, cfg = load_yamls()


def one_hot_encode(seq):
    strings_to_encode = "ACGTNX"
    strings_to_encode = "ACGT"
    mapping = dict(zip(strings_to_encode, range(len(strings_to_encode))))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]


def get_X_and_y(df_seq, chromosones):

    labelA = df_seq["label"].values
    labelB = np.logical_not(df_seq["label"].values).astype(int)
    labels = np.array([[a, b] for a, b in zip(labelA, labelB)])

    sequences = df_seq["sequences"].values

    N_chromosones = len(chromosones)

    if not (
        Path(paths["X"].format(N_chromosones=N_chromosones)).is_file()
        and Path(paths["y"].format(N_chromosones=N_chromosones)).is_file()
    ):
        one_hot_encoded_seqs = []
        for sequence in tqdm(sequences):
            one_hot_encoded_seqs.append(one_hot_encode(sequence).T)
        one_hot_encoded_seqs = np.stack(one_hot_encoded_seqs)
        one_hot_encoded_seqs.shape

        X = torch.tensor(one_hot_encoded_seqs).type(torch.float)
        y = torch.tensor(labels).view(-1, 2).type(torch.float)

        torch.save(X, paths["X"].format(N_chromosones=N_chromosones))
        torch.save(y, paths["y"].format(N_chromosones=N_chromosones))

    else:
        X = torch.load(paths["X"].format(N_chromosones=N_chromosones))
        y = torch.load(paths["y"].format(N_chromosones=N_chromosones))

    if cfg["use_gpu"]:
        X = X.cuda()
        y = y.cuda()

    return X, y


# simple loss function for batch
def simple_loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

        wandb.log({"loss": loss})

    return loss.item(), len(xb)


def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    "Computes accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid:
        y_pred = y_pred.sigmoid()
    return ((y_pred > thresh).byte() == y_true.byte()).float().mean()


# simple fit function
def simple_fit(epochs, model, loss_func, opt, train_dl, valid_dl):

    epoch_list = []
    train_loss_list = []
    valid_loss_list = []
    acc_list = []
    y_hat_list = []
    y_true_list = []

    print("epoch\ttrain loss\tvalid loss\taccuracy")

    for epoch in range(epochs):

        model.train()
        train_losses, train_nums = zip(
            *[simple_loss_batch(model, loss_func, xb, yb, opt) for xb, yb in train_dl]
        )

        # loss calculation for every epoch
        train_loss = np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
        train_loss_list.append(train_loss)

        model.eval()
        with torch.no_grad():
            valid_losses, valid_nums = zip(
                *[simple_loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )

            # calculations for accuracy_thres metric:
            y_hat = [model(xb) for xb, _ in valid_dl]
            y_true = [yb for _, yb in valid_dl]

            y_hat_list.append(torch.cat(y_hat).cpu().numpy())
            y_true_list.append(torch.cat(y_true).cpu().numpy())

        valid_loss = np.sum(np.multiply(valid_losses, valid_nums)) / np.sum(valid_nums)
        acc = accuracy_thresh(torch.cat(y_hat), torch.cat(y_true))

        epoch_list.append(epoch)
        valid_loss_list.append(valid_loss)
        acc_list.append(acc.item())

        # WandB – wandb.log(a_dict) logs the keys and values of the dictionary passed in and associates the values with a step.
        # You can log anything by passing it to wandb.log, including histograms, custom matplotlib objects, images, video, text, tables, html, pointclouds and other 3D objects.
        # Here we use it to log test accuracy, loss and some test images (along with their true and predicted labels).
        wandb.log(
            {"Epoch": epoch, "Validation Loss": valid_loss, "Validation Accuracy": acc.item()}
        )

        print(f"{epoch}\t{train_loss:.6f}\t{valid_loss:.6f}\t{acc.detach().item():.3f}")

    # print training (https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py)
    # pdb.set_trace()
    plt.subplot(2, 1, 1)
    plt.title("Training")
    plt.ylabel("Loss")
    plt.plot(train_loss_list, label="Train loss")
    plt.plot(valid_loss_list, label="Valid loss")
    plt.legend(loc=1)

    plt.subplot(2, 1, 2)
    plt.xlabel("Training batch")
    plt.ylabel("Accuracy")
    plt.plot(acc_list, label="Accuracy metric")
    plt.legend(loc=1)

    plt.show()

    # pdb.set_trace()

    return (
        epoch_list,
        train_loss_list,
        valid_loss_list,
        acc_list,
        np.concatenate(y_hat_list),
        np.concatenate(y_true_list),
    )
