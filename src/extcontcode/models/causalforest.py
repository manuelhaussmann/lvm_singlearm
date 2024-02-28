import warnings

import click
import numpy as np
import torch as th
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from tqdm import tqdm

from benchruns.exp_benchmark import load_exp_data

warnings.filterwarnings("ignore")


def collect_data(loader):
    ts = []
    Xs = []
    mu0s = []
    mu1s = []
    ys = []
    with th.no_grad():
        for x, y, t, mu0, mu1 in loader:
            t = t.flatten()
            ts.append(t)
            Xs.append(x)
            mu0s.append(mu0)
            mu1s.append(mu1)
            ys.append(y)
    ts = th.cat(ts)
    Xs = th.cat(Xs)
    ys = th.cat(ys)
    mu0s = th.cat(mu0s)
    mu1s = th.cat(mu1s)

    return Xs, ys, ts, mu0s, mu1s


def get_causal_forest():
    return CausalForestDML(
        discrete_treatment=True,
        model_t=GradientBoostingClassifier(),
        model_y=GradientBoostingRegressor(),
    )


def infer_forest(model, loader):
    Xs, ys, ts, mu0s, mu1s = collect_data(loader)
    model.fit(ys.numpy()[:, 0], ts.numpy(), X=Xs.numpy())
    return model


def get_pehe(model, loader, std_y):
    Xs, ys, ts, mu0s, mu1s = collect_data(loader)
    estimate = model.effect(Xs)
    truth = (mu1s - mu0s) / std_y
    tmp = (truth - estimate).pow(2).mean().sqrt()
    return tmp


@click.command()
@click.option("--max_id", default=1)
@click.option("--npred", default=25)
@click.option("--shift", default=0)
@click.option("--flip", default=0.0)
@click.option("--verbose", default=False, is_flag=True)
def main(max_id, npred, shift, flip, verbose):
    print("##### CAUSAL FOREST #####")
    res_train = []
    res_test = []
    for id in tqdm(range(max_id), leave=False):
        # print("ID")
        model = get_causal_forest()
        data_dict = load_exp_data(
            data_set="ihdp",
            exp_setting="full",
            params={"id": id, "npred": npred, "shift": shift, "prob_flip": flip},
        )
        infer_forest(model, data_dict["train_loader"])
        res_train.append(get_pehe(model, data_dict["train_loader"], data_dict["std_y"]))
        res_test.append(get_pehe(model, data_dict["test_loader"], data_dict["std_y"]))
    res_train = np.stack(res_train)
    res_test = np.stack(res_test)
    np.savez(
        f"causalforest_npred{npred}_shift{shift}_flip{flip}.npy",
        train=res_train,
        test=res_test,
    )

    if verbose:
        print(
            f"Train: {np.mean(res_train):.4f} \pm {np.std(res_train) / res_train.shape[0]:.4}"
        )
        print(
            f"Test: {np.mean(res_test):.4f} \pm {np.std(res_test) / res_test.shape[0]:.4}"
        )
    return res_train, res_test


if __name__ == "__main__":
    main()
