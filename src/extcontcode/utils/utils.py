import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as th
from sklearn.decomposition import PCA
from torch import distributions as thd
from tqdm import tqdm


def normcdf(x, mu, sigma):
    """
    Compute the normal CDF function via the error function. The erf allows
    for backpropagation if you want to.
    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    # Faster than going via torch
    return 0.5 * (1 + th.special.erf((x - mu) / (sigma * math.sqrt(2))))


def impute_missing(
    data,
    missing_mask,
    distributions,
    deterministic=True,
    mod_orig=False,
    ret_imputations=False,
    skip=None,
):
    """
    In the deterministic imputation:
        Use the empirical mean for continuous covariates and the median for binary ones
    Otherwise:
        Sample with the population statistics
    :param data:
    :param missing_mask:
    :param distributions:
    :param deterministic:
    :param mod_orig:
    :param skip:
    :return:
    """
    if not mod_orig:
        data = 1.0 * data
    assert data.shape[1] == len(distributions)
    assert data.shape == missing_mask.shape

    imputations = th.zeros(len(distributions))

    for i in tqdm(range(len(distributions)), leave=False):
        if skip is not None:
            if i in skip:
                continue
        elif sum(missing_mask[:, i] == 0) == 0:
            continue
        else:
            if distributions[i] == "Bernoulli" or "Categorical" in distributions[i]:
                if deterministic:
                    data[missing_mask[:, i] == 0, i] = th.median(
                        data[missing_mask[:, i] == 1, i]
                    )
                    imputations[i] = th.median(data[missing_mask[:, i] == 1, i])
                else:
                    p = sum(data[missing_mask[:, i] == 1, i])
                    data[missing_mask[:, i] == 0, i] = thd.Bernoulli(p).sample(
                        data[missing_mask[:, i] == 0, i].shape
                    )
            elif distributions[i] == "Normal":
                if deterministic:
                    data[missing_mask[:, i] == 0, i] = th.mean(
                        data[missing_mask[:, i] == 1, i]
                    )
                    imputations[i] = th.mean(data[missing_mask[:, i] == 1, i])
                else:
                    mean = data[missing_mask[:, i] == 1, i].mean()
                    std = data[missing_mask[:, i] == 1, i].std()
                    data[missing_mask[:, i] == 0, i] = thd.Normal(mean, std).sample(
                        data[missing_mask[:, i] == 0, i].shape
                    )
            else:
                raise NotImplementedError(
                    f"Sorry, input imputation is not implemented for {distributions[i]}"
                )
    if ret_imputations:
        return imputations
    else:
        return data


def fit_pca(data_train, data_pred, ndim=5, ret_pca=False):
    pca = PCA(n_components=ndim)
    pca.fit(data_train)
    if ret_pca:
        return pca, pca.transform(data_pred)
    else:
        return pca.transform(data_pred)


def print_tex_results(data, names):
    n_mods = data.shape[2]
    # order = [5, 6, 4, 1, 0, 2, 3]
    order = [4, 3, 6, 5, 1, 2, 0]

    # for n in range(n_mods):
    for n in order:
        d = data[:, :, n]
        means = d.mean(1)
        stes = d.std(1) / np.sqrt(d.shape[1])

        print(names[n])
        for i in range(data.shape[0]):
            if i == 1:
                continue
            print(f"${means[i]:.3f} \\sd" + "{\pm " + f"{stes[i]:.3f}" + "}$ &", end="")
        print("\n")


def softplus(x):
    return th.nn.Softplus().forward(x)


def violin_plots(data, names):
    order = np.array([2, 1, 4, 3, 0])

    df = pd.DataFrame(data[:, order])
    df.columns = np.array(names)[order]

    sns.violinplot(df)
    sns.despine()
    plt.show()


def turn_cat_to_binary(covars, distributions, keep_categorical=False):
    assert check_sorted_dists(distributions)
    if type(covars) == np.ndarray:
        covars = th.from_numpy(covars)
    dists = []

    restruct_covar = th.Tensor([])
    first = None
    for i, dist in enumerate(distributions):
        if "Categorical" in dist:
            if first is None:
                first = i
            n_cat = int(dist.split("_")[1])
            dists += ["Bernoulli" for _ in range(n_cat)]
            restruct_covar = th.cat(
                (restruct_covar, one_hot(covars[:, i].long(), n_cat)), 1
            )
    if first is not None:
        if keep_categorical:
            dists = distributions
        else:
            dists = distributions[:first] + dists
        restruct_covar = th.cat((covars[:, :first], restruct_covar), 1)
        return dists, restruct_covar
    else:
        return distributions, covars


def one_hot(vec, K):
    return th.eye(K)[vec]


def sort_covars(covars, loc_ber, loc_norm, loc_cat):
    return np.concatenate(
        (covars[:, loc_ber], covars[:, loc_norm], covars[:, loc_cat]), 1
    )


def create_custom_loader(tensors, n_train, n_val, n_batch=100):
    """
    Combine a couple of tensors into a combined set of train/val/test loaders
    :param tensors: list of tensors to be combined
    :param n_train: nr of training points
    :param n_val: nr of validation points
    :param n_batch: nr of data points per batch (shared between all loaders)
    :return: train,val,test loader
    """
    n_data = tensors[0].shape[0]
    indices = np.random.permutation(np.arange(n_data))
    assert (
        n_train + n_val < n_data
    ), "Can't have more training + val data than actual data"
    train_ind = indices[:n_train]
    val_ind = indices[n_train : (n_train + n_val)]
    test_ind = indices[(n_train + n_val) :]

    train_loader = th.utils.data.DataLoader(
        th.utils.data.TensorDataset(*(tensor[train_ind] for tensor in tensors)),
        batch_size=n_batch,
        shuffle=True,
    )
    val_loader = th.utils.data.DataLoader(
        th.utils.data.TensorDataset(*(tensor[val_ind] for tensor in tensors)),
        batch_size=n_batch,
        shuffle=False,
    )
    test_loader = th.utils.data.DataLoader(
        th.utils.data.TensorDataset(*(tensor[test_ind] for tensor in tensors)),
        batch_size=n_batch,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def check_sorted_dists(dists):
    n_ber = sum(["Bernoulli" in dist for dist in dists])
    n_norm = sum(["Normal" in dist for dist in dists])
    n_cat = sum(["Categorical" in dist for dist in dists])
    assert n_ber + n_norm + n_cat == len(
        dists
    ), f"The list of distributions contains unauthorized ones, see {dists}"
    if not all(["Bernoulli" in dist for dist in dists[:n_ber]]):
        return False
    if not all(["Normal" in dist for dist in dists[n_ber : (n_ber + n_norm)]]):
        return False
    if not all(["Categorical" in dist for dist in dists[(n_ber + n_norm) :]]):
        return False
    return True


def sort_dists(dists):
    """
    Sort an unsorted list of distributions into (Bernoulli, Normal, Categorical)
    :param dists: list of distributions
    :return: sorted dists
    """
    loc_ber = ["Bernoulli" in dist for dist in dists]
    loc_norm = ["Normal" in dist for dist in dists]
    loc_cat = ["Categorical" in dist for dist in dists]
    # Probably not the nicest approach, but it works, so let's all be happy
    out_dist = (
        [dists[i] for i in np.arange(len(dists))[loc_ber]]
        + [dists[i] for i in np.arange(len(dists))[loc_norm]]
        + [dists[i] for i in np.arange(len(dists))[loc_cat]]
    )
    return out_dist, loc_ber, loc_norm, loc_cat


def count_dists(distributions):
    n_ber = sum(["Bernoulli" in dist for dist in distributions])
    n_norm = sum(["Normal" in dist for dist in distributions])
    n_cat = sum(["Categorical" in dist for dist in distributions])
    n_catpred = sum(
        [int(cat.split("_")[1]) for cat in distributions if "Categorical" in cat]
    )
    return n_ber, n_norm, n_cat, n_catpred


def lvar_to_std(lvar):
    return th.exp(0.5 * lvar)
