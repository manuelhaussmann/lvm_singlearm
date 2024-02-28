# Collection of scripts for evaluating the public benchmark IHDP

import warnings

import click
import numpy as np
import torch as th
from lifelines import KaplanMeierFitter
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from tqdm import tqdm

from extcontcode.benchdata.ihdp.ihdp import load_ihdp_setting
from extcontcode.utils.distances import (
    sq_euclid,
    compute_2wasserstein_diagnormal,
    compute_sqhellinger_diagnormal,
)
from extcontcode.utils.experiments import generate_model
from extcontcode.utils.matching import (
    get_matches,
    get_subset_matched,
)
from extcontcode.utils.propensity import fit_psmodel


def collect_att_pehe(
    max_id, n_pred=25, zdim=5, shift=0, flip=0.0, setting="full", att=False, norm=True
):
    "Load predictions from a folder (assumed to be `runs/ihdp`) and compute ATT/CATE"
    res_train = []
    res_val = []
    res_test = []
    names = [
        "snet",
        "our_snet_sep",
        "our_snet_ident_sep",
        "our_snet",
        "our_snet_ident",
        "tedvae",
        "our_tedvae",
        "our_tedvae_ident",
        "our_tedvae_sep",
        "our_tedvae_ident_sep",
        "singlenet",
        "tnet",
        "ident_sep",
        "ident",
        "our_sep",
        "cfrnet",
        "tarnet",
        "our",
        "vae_y",
    ]
    for id in tqdm(range(max_id), leave=False):
        path = load_ihdp_setting(
            id=id, perc_val=0.1, n_pred=n_pred, shift_size=shift, prob_flip=flip
        )

        preds = [
            th.load(
                f"../runs/ihdp/{setting}_z{zdim}_npred{n_pred}_shift{shift}_flip{flip}_{id}/{model}/preds.pt"
            )
            for model in names
        ]

        if att:
            eval_func = eval_att
        else:
            eval_func = eval_cate
        for ldid, loader in enumerate(["train", "val", "test"]):
            res = [eval_func(pr, ldid, path, norm) for pr in preds]

            if ldid == 0:
                res_train.append(res)
            elif ldid == 1:
                res_val.append(res)
            elif ldid == 2:
                res_test.append(res)
    return names, np.stack(res_train), np.stack(res_val), np.stack(res_test)


def eval_att(model, loader, path, norm_pehe):
    "Compute ATT from precomputed predictions"
    preds = model[loader]
    true_cate = preds[:, 1] - preds[:, 0]
    pred_cate = preds[:, 4] - preds[:, 3]
    if norm_pehe:
        true_cate /= path["std_y"]
    else:
        pred_cate *= path["std_y"]
    return (
        true_cate[preds[:, 2].eq(1)].mean() - pred_cate[preds[:, 2].eq(1)].mean()
    ).abs()


def eval_cate(model, loader, path, norm_pehe):
    "Compute CATE from precomputed predictions"
    preds = model[loader]
    true_cate = preds[:, 1] - preds[:, 0]
    pred_cate = preds[:, 4] - preds[:, 3]
    if norm_pehe:
        true_cate /= path["std_y"]
    else:
        pred_cate *= path["std_y"]
    return (true_cate - pred_cate).pow(2).mean().sqrt()


def preproc_latent_space_ihdp(
    models,
    ids,
    n_pred,
    zdim,
    shift,
    flip,
    setting="partial",
    loader="train_loader",
    verbose=False,
):
    "Precompute the latent embeddings for all model"
    if verbose:
        print(
            "INFO: We are relying on a dictionary of hardcoded metaparameters in this function"
        )
    model_params = {
        "dim_x": 28,
        "dim_z": zdim,
        "dim_u": 50,
        "dim_zm": 10,
        "reg_mmd": 1.0,
        "discr": True,
        "sig_y": 0.1,
        "sig_x": 1.0,
        "lam": 1.0,
        "missing_x": False,
    }
    res = {}
    for model_name in models:
        res[model_name] = {}
        res[model_name]["tot_zs"] = []
        res[model_name]["tot_zsvar"] = []
        res[model_name]["tot_ts"] = []
        res[model_name]["tot_Xs"] = []
        res[model_name]["tot_mu0s"] = []
        res[model_name]["tot_mu1s"] = []
        res[model_name]["tot_ys"] = []
    for id in tqdm(range(ids), leave=False):
        path = load_ihdp_setting(
            id=id,
            perc_val=0.1,
            n_pred=n_pred,
            shift_size=shift,
            prob_flip=flip,
            partial="partial" == setting,
            tte="tte" == setting,
        )
        for model_name in models:
            model = generate_model(
                model_name,
                model_params,
                path["distributions"],
                c_index=path["c_index"],
                tte="tte" == setting,
            )
            model.load_state_dict(
                th.load(
                    f"../runs/ihdp/{setting}_z{model_params['dim_z']}_npred{n_pred}_shift{shift}_flip{flip}_{id}/{model_name}/model.pt",
                    map_location=th.device("cpu"),
                )
            )

            zs = []
            zsvar = []
            ts = []
            Xs = []
            mu0s = []
            mu1s = []
            ys = []
            with th.no_grad():
                for x, y, t, mu0, mu1 in path[loader]:
                    if hasattr(model, "missing_x") and model.missing_x:
                        x = x
                    else:
                        x = x[:, : model.dim_x]
                    t = t.flatten()
                    ts.append(t)
                    Xs.append(x)
                    mu0s.append(mu0)
                    mu1s.append(mu1)
                    if model_name in [
                        "our",
                        "our_sep",
                        "vae_y",
                        "vae",
                        "ident",
                        "ident_sep",
                        "our_snet",
                        "our_snet_ident",
                        "our_snet_sep",
                        "our_snet_ident_sep",
                        "tedvae",
                        "our_tedvae",
                        "our_tedvae_ident",
                        "our_tedvae_sep",
                        "our_tedvae_ident_sep",
                    ]:
                        _, zmean, zvar = model.get_qz_x(x)
                        zs.append(zmean)
                        zsvar.append(zvar)
                    elif model_name == "snet":
                        zs.append(model.get_z(x))
                        zsvar.append(th.zeros_like(zs[-1]))
                    else:
                        zs.append(model.z_nn(x))
                        zsvar.append(th.zeros_like(zs[-1]))
                    ys.append(y)
            zs = th.cat(zs)
            zsvar = th.cat(zsvar)
            ts = th.cat(ts)
            Xs = th.cat(Xs)
            ys = th.cat(ys)
            mu0s = th.cat(mu0s)
            mu1s = th.cat(mu1s)

            if setting != "tte":
                mu0s = (mu0s - path["mean_y"]) / path["std_y"]
                mu1s = (mu1s - path["mean_y"]) / path["std_y"]

            res[model_name]["tot_zs"].append(zs)
            res[model_name]["tot_zsvar"].append(zsvar)
            res[model_name]["tot_ts"].append(ts)
            res[model_name]["tot_Xs"].append(Xs)
            res[model_name]["tot_ys"].append(ys)
            res[model_name]["tot_mu0s"].append(mu0s)
            res[model_name]["tot_mu1s"].append(mu1s)
    for model_name in models:
        th.save(
            {
                "tot_zs": res[model_name]["tot_zs"],
                "tot_zsvar": res[model_name]["tot_zsvar"],
                "tot_ts": res[model_name]["tot_ts"],
                "tot_Xs": res[model_name]["tot_Xs"],
                "tot_ys": res[model_name]["tot_ys"],
                "tot_mu0s": res[model_name]["tot_mu0s"],
                "tot_mu1s": res[model_name]["tot_mu1s"],
            },
            f"../runs/{setting}_z{model_params['dim_z']}_npred{n_pred}_shift{shift}_flip{flip}_{model_name}_{loader}_precomp_latent.pt",
        )


def eval_tte(
    setting,
    ids=1000,
    npred=25,
    shift=0,
    flip=0.0,
    loader="train_loader",
    distance="euclid",
    fit_ps_obs=False,
    fit_ps_latent=False,
    fit_ps_pca=False,
):
    "Evaluate the ATT after matching in the latent space"
    models = [
        "tarnet",
        "cfrnet",
        "vae",
        "vae_y",
        "our",
        "ident",
        "our_sep",
        "ident_sep",
        "snet",
        "our_snet",
        "our_snet_ident",
    ]

    res = dict()
    for model in models:
        res[model] = []
        data = th.load(
            f"../runs/{setting}_z5_npred{npred}_shift{shift}_flip{flip}_{model}_{loader}_precomp_latent.pt"
        )
        for id in tqdm(range(ids), leave=False):
            zs = data["tot_zs"][id]
            zsvar = th.exp(0.5 * data["tot_zsvar"][id])
            ts = data["tot_ts"][id]
            Xs = data["tot_Xs"][id]
            ys = data["tot_ys"][id]
            mu0s = data["tot_mu0s"][id]
            mu1s = data["tot_mu1s"][id]

            y, ymask, ydelta = ys[:, 0], ys[:, 1], ys[:, 2]
            assert ts.shape == y.shape

            if "snet" in model:
                # We don't care about the predictability for the treatment assigment
                assert zs.shape[1] == 6
                zs = zs[:, :-1]

            if fit_ps_obs:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ps = th.from_numpy(fit_psmodel(Xs, ts)[:, None])
                distmatrix = sq_euclid(ps, ps)
            elif fit_ps_latent:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ps = th.from_numpy(fit_psmodel(zs, ts)[:, None])
                distmatrix = sq_euclid(ps, ps)
            elif fit_ps_pca:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    xpca = PCA(n_components=5).fit_transform(Xs)
                    ps = th.from_numpy(fit_psmodel(xpca, ts)[:, None])
                distmatrix = sq_euclid(ps, ps)
            elif distance == "euclid":
                distmatrix = sq_euclid(zs, zs)
            elif distance == "wasserstein":
                distmatrix = compute_2wasserstein_diagnormal(zs, zsvar, zs, zsvar)
            elif distance == "hellinger":
                distmatrix = compute_sqhellinger_diagnormal(zs, zsvar, zs, zsvar)
            else:
                raise NotImplementedError(f"{distance} is not vailable")

            matched = get_matches(distmatrix, ts)
            match_data = get_subset_matched(ys, ts.eq(1), matched)

            res[model].append((match_data, mu0s, mu1s, ys, ts))

        if fit_ps_obs or fit_ps_latent or fit_ps_pca:
            tqdm.write(
                f"ps: ${np.mean(res[model]):.3f} \\sd"
                + "{\\pm"
                + f"{np.std(res[model]) / np.sqrt(ids):.3f}"
                + "}$ &"
            )
            res
            return res

    th.save(
        res,
        f"../tte/{setting}_{loader}_z5_npred{npred}_shift{shift}_flip{flip}_matched.pt",
    )
    return res


def eval_att_partial_ihdp(
    setting,
    ids=1000,
    npred=25,
    shift=0,
    flip=0.0,
    loader="train_loader",
    distance="euclid",
    fit_ps_obs=False,
    fit_ps_latent=False,
    fit_ps_pca=False,
):
    "Evaluate the ATT after matching in the latent space"
    models = [
        "tarnet",
        "cfrnet",
        "vae",
        "vae_y",
        "our",
        "ident",
        "our_sep",
        "ident_sep",
        "snet",
        "our_snet",
        "our_snet_ident",
        "our_snet_sep",
        "our_snet_sep_ident",
        "tedvae",
        "our_tedvae",
        "our_tedvae_ident",
        "our_tedvae_sep",
        "our_tedvae_ident_sep",
    ]

    res = dict()
    for model in models:
        res[model] = []
        data = th.load(
            f"../runs/{setting}_z5_npred{npred}_shift{shift}_flip{flip}_{model}_{loader}_precomp_latent.pt"
        )
        for id in tqdm(range(ids), leave=False):
            zs = data["tot_zs"][id]
            zsvar = th.exp(0.5 * data["tot_zsvar"][id])
            ts = data["tot_ts"][id]
            Xs = data["tot_Xs"][id]
            ys = data["tot_ys"][id]
            mu0s = data["tot_mu0s"][id]
            mu1s = data["tot_mu1s"][id]

            ys = ys[:, 0]  # remove the masking dimension
            assert ts.shape == ys.shape

            if fit_ps_obs:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ps = th.from_numpy(fit_psmodel(Xs, ts)[:, None])
                distmatrix = sq_euclid(ps, ps)
            elif fit_ps_latent:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ps = th.from_numpy(fit_psmodel(zs, ts)[:, None])
                distmatrix = sq_euclid(ps, ps)
            elif fit_ps_pca:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    xpca = PCA(n_components=5).fit_transform(Xs)
                    ps = th.from_numpy(fit_psmodel(xpca, ts)[:, None])
                distmatrix = sq_euclid(ps, ps)
            elif distance == "euclid":
                distmatrix = sq_euclid(zs, zs)
            elif distance == "wasserstein":
                distmatrix = compute_2wasserstein_diagnormal(zs, zsvar, zs, zsvar)
            elif distance == "hellinger":
                distmatrix = compute_sqhellinger_diagnormal(zs, zsvar, zs, zsvar)
            else:
                raise NotImplementedError(f"{distance} is not vailable")

            linear_sum = True
            if linear_sum:
                row_ind, col_ind = linear_sum_assignment(
                    distmatrix[ts.eq(1)][:, ts.eq(0)]
                )
                match_data = [ys[ts.eq(1)][row_ind], ys[ts.eq(0)][col_ind]]
            else:
                matched = get_matches(distmatrix, ts)
                match_data = get_subset_matched(ys, ts.eq(1), matched)
            att = (
                (match_data[0].mean() - match_data[1].mean())
                - (mu1s[ts.eq(1)].mean() - mu0s[ts.eq(1)].mean())
            ).abs()
            res[model].append(att)

        if fit_ps_obs or fit_ps_latent or fit_ps_pca:
            tqdm.write(
                f"ps: ${np.mean(res[model]):.3f} \\sd"
                + "{\\pm"
                + f"{np.std(res[model]) / np.sqrt(ids):.3f}"
                + "}$ &"
            )
            res
            return res
        tqdm.write(
            f"{model}: ${np.mean(res[model]):.3f} \\sd"
            + "{\\pm"
            + f"{np.std(res[model]) / np.sqrt(ids):.3f}"
            + "}$ &"
        )
    return res


@click.command()
@click.option("--save_name", default="results")
@click.option("--max_id", default=500)
@click.option("--n_pred", default=25)
@click.option("--zdim", default=5)
@click.option("--shift", default=0)
@click.option("--flip", default=0.0)
@click.option("--setting", default="full")
@click.option("--loader", default="train_loader")
@click.option("--distance", default="euclid")
@click.option("--fit_ps_obs", default=False, is_flag=True)
@click.option("--fit_ps_latent", default=False, is_flag=True)
@click.option("--fit_ps_pca", default=False, is_flag=True)
@click.option("--pehe", default=False, is_flag=True)
@click.option("--att", default=False, is_flag=True)
@click.option("--att_partial", default=False, is_flag=True)
@click.option("--tte_partial", default=False, is_flag=True)
@click.option("--save", default=False, is_flag=True)
@click.option("--precomp", default=False, is_flag=True)
def main(
    save_name,
    max_id,
    n_pred,
    zdim,
    shift,
    flip,
    setting,
    loader,
    distance,
    fit_ps_obs,
    fit_ps_latent,
    fit_ps_pca,
    pehe,
    att,
    att_partial,
    tte_partial,
    save,
    precomp,
):
    "Evaluate a set of metrics in different settings and loaders"
    verbose = True
    save_name = (
        save_name
        + f"_{setting}_maxid{max_id}_npred{n_pred}_zdim{zdim}_shift{shift}_flip{flip}"
    )
    print(save_name)
    if pehe:
        names, *res = collect_att_pehe(
            max_id,
            n_pred=n_pred,
            zdim=zdim,
            shift=shift,
            flip=flip,
            setting=setting,
            att=False,
            norm=True,
        )
        if verbose:
            print("\n\n### PEHE")
            print(names)
            print(res[0].mean(0))
            print(res[1].mean(0))
            print(res[2].mean(0))

            print("# SE")
            print(res[0].std(0) / np.sqrt(max_id))
            print(res[1].std(0) / np.sqrt(max_id))
            print(res[2].std(0) / np.sqrt(max_id))

        if save:
            np.save(save_name + "_pehe.npy", res)
            with open(save_name + "_pehe_order.csv", "w") as f:
                f.write(",".join(names))

    if att:
        names, *res = collect_att_pehe(
            max_id,
            n_pred=n_pred,
            zdim=zdim,
            shift=shift,
            flip=flip,
            setting=setting,
            att=True,
            norm=True,
        )
        if verbose:
            print("\n\n### ATT")
            print(names)
            print(res[0].mean(0))
            print(res[1].mean(0))
            print(res[2].mean(0))

            print("# SE")
            print(res[0].std(0) / np.sqrt(max_id))
            print(res[1].std(0) / np.sqrt(max_id))
            print(res[2].std(0) / np.sqrt(max_id))

        if save:
            np.save(save_name + "_att.npy", res)
            with open(save_name + "_att_order.csv", "w") as f:
                f.write(",".join(names))

    if precomp:
        preproc_latent_space_ihdp(
            [
                "tarnet",
                "cfrnet",
                "vae",
                "our",
                "our_sep",
                "vae_y",
                "ident",
                "ident_sep",
                "snet",
                "our_snet",
                "our_snet_ident",
                "our_snet_sep",
                "our_snet_ident_sep",
                "tedvae",
                "our_tedvae",
                "our_tedvae_ident",
                "our_tedvae_sep",
                "our_tedvae_ident_sep",
            ]
            + (["vae"] if setting == "partial" else []),
            max_id,
            n_pred,
            zdim=zdim,
            shift=shift,
            flip=flip,
            setting=setting,
            loader=loader,
        )

    if att_partial:
        eval_att_partial_ihdp(
            setting=setting,
            ids=max_id,
            npred=n_pred,
            shift=shift,
            flip=flip,
            loader=loader,
            distance=distance,
            fit_ps_obs=fit_ps_obs,
            fit_ps_latent=fit_ps_latent,
            fit_ps_pca=fit_ps_pca,
        )
    if tte_partial:
        eval_tte(
            setting=setting,
            ids=max_id,
            npred=n_pred,
            shift=shift,
            flip=flip,
            loader=loader,
            distance=distance,
            fit_ps_obs=fit_ps_obs,
            fit_ps_latent=fit_ps_latent,
            fit_ps_pca=fit_ps_pca,
        )


if __name__ == "__main__":
    main()
