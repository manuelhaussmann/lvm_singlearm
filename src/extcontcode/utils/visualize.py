import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as th
from sklearn.manifold import TSNE
from tqdm import tqdm

from extcontcode.utils.balancemetrics import standardized_diff


def vis_2d(X, title="Latent Embedding", indset=None):
    if X.shape[1] == 2:
        X2d = X
    else:
        tsne = TSNE(n_components=2)
        X2d = tsne.fit_transform(X)

    if indset is None:
        plt.scatter(X2d[:, 0], X[:, 1], alpha=0.7)
    else:
        for i in np.unique(indset):
            plt.scatter(X2d[indset.eq(i), 0], X[indset.eq(i), 1], alpha=0.7)
    plt.title(title)
    plt.show()


def scatter(X, filter):
    for i in np.unique(filter):
        plt.scatter(X[filter.eq(i), 0], X[filter.eq(i), 1])
    plt.show()


def visualize_curves(curves, names, ylogscale=True):
    assert len(curves) == len(names)
    for curve, name in zip(curves, names):
        plt.plot(curve, label=name)
    if ylogscale:
        plt.yscale("log")
    plt.legend()
    plt.show()


def map_vis_tsne(data, pop, title="", save_str=None):
    mapped = TSNE(n_components=2).fit_transform(data)
    vis_two_pops(mapped, pop, ["RWD", "RCT"], title, save_str)
    return mapped


def vis_two_pops(data, pop, label=["RWD", "RCT"], title="", save_str=None):
    plt.scatter(data[pop.eq(0), 0], data[pop.eq(0), 1], alpha=0.4, label=label[0])
    plt.scatter(data[pop.eq(1), 0], data[pop.eq(1), 1], alpha=0.4, label=label[1])
    plt.legend()
    plt.title(title)
    if save_str is not None:
        plt.savefig(save_str)
        plt.close()
    else:
        plt.show()


def distplot(data, label=None, set_theme=True):
    if set_theme:
        sns.set_theme()

    sns.histplot(data, kde=True, stat="density", label=label)
    sns.rugplot(data)


def get_overlaps(file):
    try:
        data = th.load(file)
    except FileNotFoundError:
        return []
    overlaps = []
    for i in tqdm(range(len(data)), leave=False):
        loader = data[i][0]
        xs = []
        ts = []
        for x, _, t, *_ in loader:
            xs.append(x)
            ts.append(t)
        xs = th.cat(xs)
        ts = th.cat(ts).flatten()
        overlaps.append(standardized_diff(xs[ts.eq(0)], xs[ts.eq(1)]))
    return overlaps


# Note: Everything is hardcoded for now
def vis_overlap(npred=5, pathdata="data/IHDP/train_val_splits", pathres="runs"):
    shifts = [0, 1, 2, 3]
    prob_flips = [0.0, 0.5, 0.6, 0.7, 0.8]

    metares = dict()

    res = []
    for sh in tqdm(shifts, leave=False):
        for pr in tqdm(prob_flips, leave=False):
            if os.path.exists(
                f"{pathres}/result_full_maxid200_npred{npred}_zdim5_shift{sh}_flip{pr}_pehe.npy"
            ):
                overlaps = np.mean(
                    get_overlaps(
                        f"{pathdata}/full_pval0.1_npred{npred}_shift{sh}_prob_flip{pr}.pt"
                    )[:-1]
                )
                performance = np.load(
                    f"{pathres}/result_full_maxid200_npred{npred}_zdim5_shift{sh}_flip{pr}_pehe.npy"
                )[
                    0
                ]  # just plot the training set for now
                res.append((overlaps, performance.mean(0), performance.std(0)))
            else:
                print(
                    "The following file does not exist:"
                    + f"{pathres}/result_full_maxid200_npred{npred}_zdim5_shift{sh}_flip{pr}_pehe.npy"
                )

    metares[f"{npred}"] = res

    X = np.array([r[0] for r in res])
    Y = np.stack([r[1] for r in res])
    Z = np.stack([r[2] for r in res]) / np.sqrt(200)

    sns.set(font_scale=2, style="white")
    names = ["our_sep+sep", "det+r", "det", "our_sep", "vae+y"]

    for i in [2, 1, 4, 3, 0]:
        order = np.argsort(X)
        plt.fill_between(
            X[order], Y[order, i] - Z[order, i], Y[order, i] + Z[order, i], alpha=0.1
        )
        plt.plot(X[order], Y[order, i], label=names[i])
    plt.legend()
    sns.despine()
    plt.xlabel("SAMPD")
    plt.xlim((X.min(), X.max()))
    plt.yscale("log")
    plt.ylabel("RMSE of CATE Estimation")
    plt.title(f"{npred}")
    plt.show()
