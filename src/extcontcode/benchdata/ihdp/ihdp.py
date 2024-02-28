# Script to preprocess the data

import os

import click
import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm

from extcontcode.benchdata.ihdp.synthetic import create_synthetic_outcome_ihdp
from extcontcode.benchdata.ihdp.tte import create_synthetic_outcome_ihdp_tte
from extcontcode.utils.utils import turn_cat_to_binary, sort_covars, sort_dists


def load_data():
    """
    Load the original IHDP data
    :return: data set, continuous, and discrete covariates
    """
    # Raw data and list of discrete/cont covariates
    data = pd.read_csv("../data/ihdp/raw_ihdp.csv")
    data = data.drop(columns="Unnamed: 0")
    assert len(data) == 747
    assert len(data.columns) == 26  # 25 covariates + one treatment variable
    covs_cont = ["bw", "b.head", "preterm", "birth.o", "nnhealth", "momage"]
    covs_cat = [
        "sex",
        "twin",
        "b.marr",
        "mom.lths",
        "mom.hs",
        "mom.scoll",
        "cig",
        "first",
        "booze",
        "drugs",
        "work.dur",
        "prenatal",
        "ark",
        "ein",
        "har",
        "mia",
        "pen",
        "tex",
        "was",
    ]
    return data, covs_cont, covs_cat


def normalize_data(df, cont_features):
    df_norm = df.copy()
    for name in cont_features:
        df_norm[name] = (df[name] - df[name].mean()) / df[name].std()
    df_norm[
        "first"
    ] -= 1  # for some reason this binary variable is shifted in the original data
    return df_norm


def ihdp_1000(
    id=0,
    perc_val=0.1,
    batch_size=100,
    n_pred=25,
    shift_size=0,
    prob_flip=0.0,
    missing=False,
    tte=False,
    partial=False,
    model_disc=True,
    verbose=False,
):
    """
    Get the 1000 precomputed train/test splits with optinally missing values.
    :param id: ID of the 1_000 replications
    :param perc_val: relative size of the validation set
    :param batch_size: batch_size
    :param n_pred: number of predictive covariates
    :param partial: missing outcome in the treatment arm
    :param model_disc: does the model treat discrete covariates differently?
    :param verbose: give me more print statements!
    :return: train, {val,}, test
    """
    train_raw = np.load("../data/IHDP/ihdp_npci_1-1000.train.npz")
    test_raw = np.load("../data/IHDP/ihdp_npci_1-1000.test.npz")

    val = perc_val != 0.0

    # There should be 672 observations in the training set or something went wrong
    assert (
        len(train_raw["t"][:, id]) == 672
    ), f"Sorry, you gave me {len(train_raw['t'][:, id])} instead of 672, which is what I asked for"

    if model_disc:
        distributions = (
            ["Normal"] * 3 + ["Categorical_4"] + ["Normal"] * 2 + ["Bernoulli"] * 19
        )
    else:
        distributions = ["Normal"] * 25

    # Get the distributions locations of the individual covariates
    distributions, loc_ber, loc_norm, loc_cat = sort_dists(distributions)
    train_x_orig = train_raw["x"][:, :, id]
    test_x = test_raw["x"][:, :, id]
    # This encoding is shifted in the original for some reason
    train_x_orig[:, 13] -= 1
    test_x[:, 13] -= 1

    cat = train_x_orig[:, 3]
    uniq_vals = np.unique(cat)
    assert (
        len(uniq_vals) == 4
    ), f"There should be max 4 classes instead of {len(uniq_vals)}"
    for i in range(4):
        train_x_orig[:, 3][cat == uniq_vals[i]] = i

    cat = test_x[:, 3]
    for i in range(4):
        test_x[:, 3][cat == uniq_vals[i]] = i

    # Sort in the categorical setup
    train_x_orig = th.from_numpy(sort_covars(train_x_orig, loc_ber, loc_norm, loc_cat))
    test_x = th.from_numpy(sort_covars(test_x, loc_ber, loc_norm, loc_cat))

    if val:
        n_val = int(672 * perc_val)
        indices = np.random.permutation(np.arange(672))
        ind_val, ind_train = indices[:n_val], indices[n_val:]
    else:
        ind_train = np.arange(672)

    train_t = th.from_numpy(train_raw["t"][ind_train, id, None])
    train_x = train_x_orig[ind_train]

    # Prepare the synthetic training outcome
    if tte:
        (
            train_x,
            Y,
            M0,
            M1,
            beta,
            ids,
            shift,
            prob_flip,
            ind_pred,
            x_missing,
            delta,
        ) = create_synthetic_outcome_ihdp_tte(
            X=train_x,
            pop=train_t.flatten(),
            n_pred=n_pred,
            shift_size=shift_size,
            prob_flip=prob_flip,
            missing=missing,
        )
    else:
        (
            train_x,
            Y,
            M0,
            M1,
            beta,
            ids,
            shift,
            prob_flip,
            ind_pred,
            x_missing,
        ) = create_synthetic_outcome_ihdp(
            X=train_x,
            pop=train_t.flatten(),
            n_pred=n_pred,
            shift_size=shift_size,
            prob_flip=prob_flip,
            missing=missing,
        )
    train_y = Y
    train_mu0 = M0
    train_mu1 = M1

    if model_disc:
        n_ber = sum(loc_ber)
        n_norm = sum(loc_norm)
        miss_cont = x_missing[:, n_ber : (n_ber + n_norm)]
        mean_x = (train_x[:, n_ber : (n_ber + n_norm)] * miss_cont).sum(
            0
        ) / miss_cont.sum(0)
        mean2_x = (train_x[:, n_ber : (n_ber + n_norm)] * miss_cont).pow(2).sum(
            0
        ) / miss_cont.sum(0)

        std_x = (mean2_x - mean_x.pow(2)).sqrt()
        train_x[:, n_ber : (n_ber + n_norm)] = (
            train_x[:, n_ber : (n_ber + n_norm)] - mean_x
        ) / std_x

    else:
        if verbose:
            print(
                "INFO: Note, I just normalized all features to mean,std=0,1 without caring about categorical or not"
            )
        mean_x = train_x.mean(0)
        std_x = train_x.std(0)
        train_x = (train_x - mean_x) / std_x

    train_y_mask = th.ones_like(train_y)
    if partial or tte:
        train_y_mask[train_t.eq(1)] = 0

    if tte:
        max_y = train_y.max()
        train_y = train_y / max_y + 0.001
        train_y = th.cat((train_y, train_y_mask, delta), 1)
        train_mu0[:, 0] /= max_y
        train_mu1[:, 0] /= max_y
    else:
        mean_y = (train_y * train_y_mask).sum() / train_y_mask.sum()
        mean2_y = (train_y * train_y_mask).pow(2).sum() / train_y_mask.sum()

        std_y = (mean2_y - mean_y.pow(2)).sqrt()
        train_y = (train_y - mean_y) / std_y
        train_y = th.cat((train_y, train_y_mask), 1)

    # shift categorical to binary encoding
    _, train_x = turn_cat_to_binary(train_x, distributions, keep_categorical=True)

    if missing:
        train_x = th.cat((train_x, x_missing), 1)

    train_list = (
        train_x,
        train_y,
        train_t,
        train_mu0,
        train_mu1,
    )

    train_loader = th.utils.data.DataLoader(
        th.utils.data.TensorDataset(*(tensor.float() for tensor in train_list)),
        batch_size=batch_size,
        shuffle=True,
    )

    if val:
        val_y = th.from_numpy(train_raw["yf"][ind_val, id, None])
        val_t = th.from_numpy(train_raw["t"][ind_val, id, None])
        val_x = train_x_orig[ind_val]

        val_y_missing = th.ones_like(val_y)
        if partial or tte:
            val_y_missing[val_t.eq(1)] = 0

        # Prepare the synthetic val outcome
        if tte:
            (
                val_x,
                Y,
                M0,
                M1,
                *_,
                val_x_missing,
                delta,
            ) = create_synthetic_outcome_ihdp_tte(
                val_x,
                val_t.flatten(),
                beta=beta,
                ids=ids,
                shift=shift,
                ind_pred=ind_pred,
                n_pred=n_pred,
                prob_flip=prob_flip,
                shift_size=shift_size,
                missing=missing,
            )
        else:
            val_x, Y, M0, M1, *_, val_x_missing = create_synthetic_outcome_ihdp(
                val_x,
                val_t.flatten(),
                beta=beta,
                ids=ids,
                shift=shift,
                ind_pred=ind_pred,
                n_pred=n_pred,
                prob_flip=prob_flip,
                shift_size=shift_size,
                missing=missing,
            )
        val_y = Y
        val_mu0 = M0
        val_mu1 = M1

        if model_disc:
            val_x[:, n_ber : (n_ber + n_norm)] = (
                val_x[:, n_ber : (n_ber + n_norm)] - mean_x
            ) / std_x
        else:
            val_x = (val_x - mean_x) / std_x

    else:
        val_mu1 = th.Tensor([th.nan])
        val_mu0 = th.Tensor([th.nan])
        val_y = th.Tensor([th.nan])
        val_y_missing = th.Tensor([th.nan])
        val_t = th.Tensor([th.nan])
        val_x = th.Tensor([th.nan])

    _, val_x = turn_cat_to_binary(val_x, distributions, keep_categorical=True)

    if missing:
        val_x = th.cat((val_x, val_x_missing), 1)

    if tte:
        val_y = val_y / max_y + 0.001
        val_y = th.cat((val_y, val_y_missing, delta), 1)
        val_mu0[:, 0] /= max_y
        val_mu1[:, 0] /= max_y
    else:
        val_y = (val_y - mean_y) / std_y
        val_y = th.cat((val_y, val_y_missing), 1)

    val_list = (
        val_x,
        val_y,
        val_t,
        val_mu0,
        val_mu1,
    )

    val_loader = th.utils.data.DataLoader(
        th.utils.data.TensorDataset(*(tensor.float() for tensor in val_list)),
        batch_size=batch_size,
        shuffle=False,
    )

    test_t = th.from_numpy(test_raw["t"][:, id, None])

    # Prepare the synthetic test outcome
    if tte:
        (
            test_x,
            Y,
            M0,
            M1,
            *_,
            test_x_missing,
            delta,
        ) = create_synthetic_outcome_ihdp_tte(
            test_x,
            test_t.flatten(),
            beta=beta,
            ids=ids,
            shift=shift,
            ind_pred=ind_pred,
            n_pred=n_pred,
            prob_flip=prob_flip,
            shift_size=shift_size,
        )
    else:
        test_x, Y, M0, M1, *_, test_x_missing = create_synthetic_outcome_ihdp(
            test_x,
            test_t.flatten(),
            beta=beta,
            ids=ids,
            shift=shift,
            ind_pred=ind_pred,
            n_pred=n_pred,
            prob_flip=prob_flip,
            shift_size=shift_size,
        )
    test_y = Y
    test_mu0 = M0
    test_mu1 = M1

    if model_disc:
        test_x[:, n_ber : (n_ber + n_norm)] = (
            test_x[:, n_ber : (n_ber + n_norm)] - mean_x
        ) / std_x
    else:
        test_x = (test_x - mean_x) / std_x

    test_y_missing = th.ones_like(test_y)
    if tte:
        test_y = test_y / max_y + 0.001
        test_y = th.cat((test_y, test_y_missing, delta), 1)
        test_mu0[:, 0] /= max_y
        test_mu1[:, 0] /= max_y
    else:
        test_y = (test_y - mean_y) / std_y
        test_y = th.cat((test_y, test_y_missing), 1)

    if partial or tte:
        test_y_missing[test_t.eq(1)] = 0
    _, test_x = turn_cat_to_binary(test_x, distributions, keep_categorical=True)

    if missing:
        test_x = th.cat((test_x, test_x_missing), 1)

    test_list = (
        test_x,
        test_y,
        test_t,
        test_mu0,
        test_mu1,
    )

    test_loader = th.utils.data.DataLoader(
        th.utils.data.TensorDataset(*(tensor.float() for tensor in test_list)),
        batch_size=batch_size,
        shuffle=False,
    )

    N_data = [
        len(train_x),
        sum(train_t == 0),
        sum(train_t == 1),
        len(val_x),
        sum(val_t == 0),
        sum(val_t == 1),
        len(test_x),
        sum(test_t == 0),
        sum(test_t == 1),
    ]
    N_data = {
        "train": [
            len(train_x),
            sum(train_t == 0).squeeze(),
            sum(train_t == 1).squeeze(),
        ],
        "val": [
            len(val_x),
            sum(val_t == 0).squeeze(),
            sum(val_t == 1).squeeze(),
        ],
        "test": [
            len(test_x),
            sum(test_t == 0).squeeze(),
            sum(test_t == 1).squeeze(),
        ],
    }

    c_index = np.random.choice(19, 3, replace=False)
    if tte:
        return (
            train_loader,
            val_loader,
            test_loader,
            max_y,
            mean_x,
            std_x,
            distributions,
            N_data,
            c_index,
        )
    else:
        return (
            train_loader,
            val_loader,
            test_loader,
            mean_y,
            std_y,
            mean_x,
            std_x,
            distributions,
            N_data,
            c_index,
        )


@click.command()
@click.option("--subset", default=1000)
@click.option("--perc_val", default=0.1)
@click.option("--batch_size", default=100)
@click.option("--n_pred", default=25)
@click.option("--shift_size", default=0)
@click.option("--prob_flip", default=0.0)
@click.option("--missing", default=False, is_flag=True)
@click.option("--tte", default=False, is_flag=True)
@click.option("--partial", default=False, is_flag=True)
@click.option("--verbose", default=False, is_flag=True)
@click.option("--path", default="../data/IHDP/train_val_splits")
def gen_ihdp_train_val_splits(
    subset=1000,
    perc_val=0.1,
    batch_size=100,
    n_pred=25,
    shift_size=0,
    prob_flip=0.0,
    missing=False,
    tte=False,
    partial=False,
    verbose=False,
    path="../data/IHDP/train_val_splits",
):
    """
    Precompute the train/val/test splits over a series over a subset of iterations
    """
    collect = dict()
    for id in tqdm(range(subset), leave=False):
        tmp = ihdp_1000(
            id,
            perc_val=perc_val,
            batch_size=batch_size,
            n_pred=n_pred,
            shift_size=shift_size,
            prob_flip=prob_flip,
            missing=missing,
            tte=tte,
            partial=partial,
            verbose=verbose,
        )
        collect[id] = tmp
    th.save(
        collect,
        f"{path}/{'tte' if tte else ('partial' if partial else 'full')}_pval{perc_val}_npred{n_pred}_shift{shift_size}_prob_flip{prob_flip}.pt",
    )
    if verbose:
        print(
            "Successfully generated the data and saved them at: "
            + f"{path}/{'tte' if tte else ('partial' if partial else 'full')}_pval{perc_val}_npred{n_pred}_shift{shift_size}_prob_flip{prob_flip}.pt",
        )


def load_ihdp_setting(
    id=1,
    perc_val=0.1,
    n_pred=25,
    shift_size=0,
    prob_flip=0.0,
    partial=False,
    tte=False,
    path="../data/IHDP/train_val_splits",
):
    """
    Load and return a precomputed IHDP setting
    """
    assert os.path.isfile(
        f"{path}/{'tte' if tte else ('partial' if partial else 'full')}_pval{perc_val}_npred{n_pred}_shift{shift_size}_prob_flip{prob_flip}.pt",
    ), (
        "This setting has not been prepared yet"
        + f"{path}/{'tte' if tte else ('partial' if partial else 'full')}_pval{perc_val}_npred{n_pred}_shift{shift_size}_prob_flip{prob_flip}.pt",
    )

    source = th.load(
        f"{path}/{'tte' if tte else ('partial' if partial else 'full')}_pval{perc_val}_npred{n_pred}_shift{shift_size}_prob_flip{prob_flip}.pt",
    )

    if tte:
        # train_loader, val_loader, test_loader, mean_y, std_y, mean_x, std_x, distributions, N_data,
        return {
            "train_loader": source[id][0],
            "val_loader": source[id][1],
            "test_loader": source[id][2],
            "max_y": source[id][3],
            "mean_x": source[id][4],
            "std_x": source[id][5],
            "distributions": source[id][6],
            "N_dat": source[id][7],
            "c_index": source[id][8],
        }
    else:
        # train_loader, val_loader, test_loader, mean_y, std_y, mean_x, std_x, distributions, N_data,
        return {
            "train_loader": source[id][0],
            "val_loader": source[id][1],
            "test_loader": source[id][2],
            "mean_y": source[id][3],
            "std_y": source[id][4],
            "mean_x": source[id][5],
            "std_x": source[id][6],
            "distributions": source[id][7],
            "N_dat": source[id][8],
            "c_index": source[id][9],
        }


if __name__ == "__main__":
    gen_ihdp_train_val_splits()
