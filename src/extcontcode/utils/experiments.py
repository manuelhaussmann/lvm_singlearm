import os

import numpy as np
import torch as th
import torch.nn as nn
from tqdm import tqdm

from extcontcode.models.detpredictors import SingleNet, TNet
from extcontcode.models.generalmissing import GeneralMissing
from extcontcode.models.generalsetup import GeneralModel
from extcontcode.models.generaltedvae import GeneralTEDVAE, IdentTEDVAE
from extcontcode.models.plainvae import PlainVAE
from extcontcode.models.tedvae import TEDVAE
from extcontcode.models.tte.detpredictors import TARNetTTE, CFRNetTTE, SNetTTE
from extcontcode.models.tte.generalident import GeneralIdentTTE
from extcontcode.models.tte.generalsnet import GeneralSNetTTE, IdentSNetTTE
from extcontcode.models.tte.generaltimetoevent import GeneralTTE
from extcontcode.models.tte.noseparate import NoUModelTTE
from extcontcode.utils.architectures import gen_fcnet


def generate_model(
        model_name,
        model_params,
        distributions,
        c_index,
        dim_y=1,
        sig_u=1.0,
        sig_z=1.0,
        beta=1.0,
        tte=False,
        device="cpu",
):
    """
    Generate a model with the specific architectural settings
    """

    if tte:
        assert not model_params["missing_x"]
        TARNet = TARNetTTE
        CFRNet = CFRNetTTE
        SNet = SNetTTE
        GeneralIdent = GeneralIdentTTE
        GeneralSNet = GeneralSNetTTE
        IdentSNet = IdentSNetTTE
        NoUModel = NoUModelTTE
        General = GeneralTTE
        dim_x = model_params["dim_x"]

    elif model_params["missing_x"] and model_name in ["our_sep", "our"]:
        General = GeneralMissing
        dim_x = 2 * model_params["dim_x"]
    else:
        General = GeneralModel
        dim_x = model_params["dim_x"]

    arch_params = {
        "f_z_rzx": gen_fcnet([model_params["dim_z"], 50]),
        "f_u0_ru0": gen_fcnet([model_params["dim_u"], 150]),
        "f_u1_ru1": gen_fcnet([model_params["dim_u"], 150]),
        "f_ruz_x": nn.Sequential(
            nn.ELU(),
            *gen_fcnet([200, 200, dim_x]),
        ),
        "f_z_x": gen_fcnet([model_params["dim_z"], 200, 200, dim_x]),
        "f_z_y0": gen_fcnet([model_params["dim_z"], 100, 100, 1]),
        "f_z_y1": gen_fcnet([model_params["dim_z"], 100, 100, 1]),
        "f_z_t": gen_fcnet([model_params["dim_z"], 1]),
        "g_x_z": gen_fcnet([dim_x, 200, 200, 2 * model_params["dim_z"]]),
        "g_x0_u0": gen_fcnet([dim_x, 200, 200, 2 * model_params["dim_u"]]),
        "g_x1_u1": gen_fcnet([dim_x, 200, 200, 2 * model_params["dim_u"]]),
        "f_zzmu_m": gen_fcnet(
            [
                model_params["dim_u"] + 10 + model_params["dim_z"],
                200,
                200,
                int(dim_x / 2),
            ]
        ),
        "g_m_zm": gen_fcnet([int(dim_x / 2), 200, 200, 2 * model_params["dim_zm"]]),
        "det_z": gen_fcnet([dim_x, 200, 200, model_params["dim_z"]]),
        "det_y0": gen_fcnet([model_params["dim_z"], 100, 100, 1]),
        "det_y1": gen_fcnet([model_params["dim_z"], 100, 100, 1]),
        "f_c_z": gen_fcnet([3, 10, 10, model_params["dim_z"]]),
        "single_y": gen_fcnet(
            [dim_x + 1, 200, 200, model_params["dim_z"], 100, 100, 1]
        ),
    }

    if model_name == "singlenet":
        model = SingleNet(y_nn=arch_params["single_y"], dim_x=dim_x)
    elif model_name == "snet":
        assert (
                model_params["dim_z"] == 5
        ), "Dimensionalities are so far hardcoded for that dim"
        model = SNet(
            z0=gen_fcnet([dim_x, 25, 25, 1]),
            z01=gen_fcnet([dim_x, 50, 50, 1]),
            z1=gen_fcnet([dim_x, 25, 25, 1]),
            za=gen_fcnet([dim_x, 100, 100, 2]),
            zt=gen_fcnet([dim_x, 100, 100, 1]),
            mu0=gen_fcnet([4, 100, 100, 1]),
            mu1=gen_fcnet([4, 100, 100, 1]),
            pi=gen_fcnet([3, 100, 100, 1]),
            dim_x=dim_x,
            dim_z=model_params["dim_z"],
        )

    elif model_name == "tnet":
        model = TNet(
            y0_nn=nn.Sequential(
                *arch_params["det_z"], nn.ELU(), *arch_params["det_y0"]
            ),
            y1_nn=nn.Sequential(
                *arch_params["det_z"], nn.ELU(), *arch_params["det_y1"]
            ),
            dim_x=dim_x,
        )
    elif model_name == "tarnet":
        model = TARNet(
            z_nn=arch_params["det_z"],
            y0_nn=arch_params["det_y0"],
            y1_nn=arch_params["det_y1"],
            dim_x=dim_x,
            dim_z=model_params["dim_z"],
        )
    elif model_name == "cfrnet":
        model = CFRNet(
            z_nn=arch_params["det_z"],
            y0_nn=arch_params["det_y0"],
            y1_nn=arch_params["det_y1"],
            reg_mmd=model_params["reg_mmd"],
            dim_x=dim_x,
            dim_z=model_params["dim_z"],
        )
    elif model_name == "tedvae":
        model = TEDVAE(
            f_z_x=arch_params["f_z_x"],
            f_z_y0=gen_fcnet([4, 100, 100, 1]),
            f_z_y1=gen_fcnet([4, 100, 100, 1]),
            f_z_t=gen_fcnet([3, 100, 100, 1]),
            g_x_zt=gen_fcnet([dim_x, 25, 25, 2 * 1]),
            g_x_zc=gen_fcnet([dim_x, 50, 50, 2 * 2]),
            g_x_zy=gen_fcnet([dim_x, 25, 25, 2 * 2]),
            dim_x0=dim_x,
            dim_x1=dim_x,
            dim_y=dim_y,
            dim_z=model_params["dim_z"],
            sig_x=model_params["sig_x"],
            sig_y=model_params["sig_y"],
            sig_z=sig_z,
            reg_mmd=model_params["reg_mmd"],
            lam=model_params["lam"],
            discriminator=model_params["discr"],
            distributions=distributions,
            probit=False,
            heterox=False,
            heteroy=False,
            beta=beta,
            detach_prop=False,
            device=device,
        )

    elif model_name in ["our_tedvae", "our_tedvae_sep"]:
        assert (
                model_params["dim_z"] == 5
        ), "Dimensionalities are so far hardcoded for that dim"
        model = GeneralTEDVAE(
            f_z_rzx=gen_fcnet([model_params["dim_z"], 50]),
            f_u0_ru0=arch_params["f_u0_ru0"],
            f_u1_ru1=arch_params["f_u1_ru1"],
            f_ruz_x=arch_params["f_ruz_x"],
            f_z_y0=gen_fcnet([4, 100, 100, 1]),
            f_z_y1=gen_fcnet([4, 100, 100, 1]),
            f_z_t=gen_fcnet([3, 100, 100, 1]),
            g_x_zt=gen_fcnet([dim_x, 25, 25, 2 * 1]),
            g_x_zc=gen_fcnet([dim_x, 50, 50, 2 * 2]),
            g_x_zy=gen_fcnet([dim_x, 25, 25, 2 * 2]),
            g_x0_u0=arch_params["g_x0_u0"],
            g_x1_u1=arch_params["g_x1_u1"],
            g_m_zm=arch_params["g_m_zm"],
            f_zzmu_m=arch_params["f_zzmu_m"],
            dim_u0=model_params["dim_u"],
            dim_u1=model_params["dim_u"],
            dim_x0=dim_x,
            dim_x1=dim_x,
            dim_y=dim_y,
            dim_z=model_params["dim_z"],
            dim_zm=model_params["dim_zm"],
            sig_u=sig_u,
            sig_x=model_params["sig_x"],
            sig_y=model_params["sig_y"],
            sig_z=sig_z,
            distributions=distributions,
            probit=False,
            heterox=False,
            heteroy=False,
            beta=beta,
            missing_x=model_params["missing_x"],
            detach_prop=False,
            sharedlatent=model_name == "our_tedvae",
            device=device,
        )
    elif model_name in ["our_tedvae_ident", "our_tedvae_ident_sep"]:
        assert (
                model_params["dim_z"] == 5
        ), "Dimensionalities are so far hardcoded for that dim"
        model = IdentTEDVAE(
            f_z_rzx=gen_fcnet([model_params["dim_z"], 50]),
            f_u0_ru0=arch_params["f_u0_ru0"],
            f_u1_ru1=arch_params["f_u1_ru1"],
            f_ruz_x=arch_params["f_ruz_x"],
            f_z_y0=gen_fcnet([4, 100, 100, 1]),
            f_z_y1=gen_fcnet([4, 100, 100, 1]),
            f_z_t=gen_fcnet([3, 100, 100, 1]),
            g_x_zt=gen_fcnet([dim_x, 25, 25, 2 * 1]),
            g_x_zc=gen_fcnet([dim_x, 50, 50, 2 * 2]),
            g_x_zy=gen_fcnet([dim_x, 25, 25, 2 * 2]),
            g_x0_u0=arch_params["g_x0_u0"],
            g_x1_u1=arch_params["g_x1_u1"],
            g_m_zm=arch_params["g_m_zm"],
            f_zzmu_m=arch_params["f_zzmu_m"],
            f_c_z=arch_params["f_c_z"],
            dim_u0=model_params["dim_u"],
            dim_u1=model_params["dim_u"],
            dim_x0=dim_x,
            dim_x1=dim_x,
            dim_y=dim_y,
            dim_z=model_params["dim_z"],
            dim_zm=model_params["dim_zm"],
            sig_u=sig_u,
            sig_x=model_params["sig_x"],
            sig_y=model_params["sig_y"],
            sig_z=sig_z,
            distributions=distributions,
            probit=False,
            heterox=False,
            heteroy=False,
            beta=beta,
            missing_x=model_params["missing_x"],
            detach_prop=False,
            c_index=c_index,
            sharedlatent=model_name == "our_tedvae_ident",
            device=device,
        )
    elif model_name in ["our_snet", "our_snet_sep"]:
        assert (
                model_params["dim_z"] == 5
        ), "Dimensionalities are so far hardcoded for that dim"
        model = GeneralSNet(
            f_z_rzx=gen_fcnet([model_params["dim_z"] + 1, 50]),
            f_u0_ru0=arch_params["f_u0_ru0"],
            f_u1_ru1=arch_params["f_u1_ru1"],
            f_ruz_x=arch_params["f_ruz_x"],
            f_z_y0=gen_fcnet([4, 100, 100, 1]),
            f_z_y1=gen_fcnet([4, 100, 100, 1]),
            f_z_t=gen_fcnet([3, 100, 100, 1]),
            g_x_z0=gen_fcnet([dim_x, 25, 25, 2 * 1]),
            g_x_z01=gen_fcnet([dim_x, 50, 50, 2 * 1]),
            g_x_z1=gen_fcnet([dim_x, 25, 25, 2 * 1]),
            g_x_za=gen_fcnet([dim_x, 100, 100, 2 * 2]),
            g_x_zt=gen_fcnet([dim_x, 100, 100, 2 * 1]),
            g_x0_u0=arch_params["g_x0_u0"],
            g_x1_u1=arch_params["g_x1_u1"],
            g_m_zm=arch_params["g_m_zm"],
            f_zzmu_m=arch_params["f_zzmu_m"],
            dim_u0=model_params["dim_u"],
            dim_u1=model_params["dim_u"],
            dim_x0=dim_x,
            dim_x1=dim_x,
            dim_y=dim_y,
            dim_z=model_params["dim_z"] + 1,
            dim_zm=model_params["dim_zm"],
            sig_u=sig_u,
            sig_x=model_params["sig_x"],
            sig_y=model_params["sig_y"],
            sig_z=sig_z,
            distributions=distributions,
            probit=False,
            heterox=False,
            heteroy=False,
            beta=beta,
            missing_x=model_params["missing_x"],
            detach_prop=False,
            sharedlatent=model_name == "our_snet",
            device=device,
        )
    elif model_name in ["our_snet_ident", "our_snet_ident_sep"]:
        assert (
                model_params["dim_z"] == 5
        ), "Dimensionalities are so far hardcoded for that dim"

        model = IdentSNet(
            f_z_rzx=gen_fcnet([model_params["dim_z"] + 1, 50]),
            f_u0_ru0=arch_params["f_u0_ru0"],
            f_u1_ru1=arch_params["f_u1_ru1"],
            f_ruz_x=arch_params["f_ruz_x"],
            f_z_y0=gen_fcnet([4, 100, 100, 1]),
            f_z_y1=gen_fcnet([4, 100, 100, 1]),
            f_z_t=gen_fcnet([3, 100, 100, 1]),
            g_x_z0=gen_fcnet([dim_x, 25, 25, 2 * 1]),
            g_x_z01=gen_fcnet([dim_x, 50, 50, 2 * 1]),
            g_x_z1=gen_fcnet([dim_x, 25, 25, 2 * 1]),
            g_x_za=gen_fcnet([dim_x, 100, 100, 2 * 2]),
            g_x_zt=gen_fcnet([dim_x, 100, 100, 2 * 1]),
            g_x0_u0=arch_params["g_x0_u0"],
            g_x1_u1=arch_params["g_x1_u1"],
            g_m_zm=arch_params["g_m_zm"],
            f_zzmu_m=arch_params["f_zzmu_m"],
            f_c_z=gen_fcnet([3, 10, 10, model_params["dim_z"] + 1]),
            dim_u0=model_params["dim_u"],
            dim_u1=model_params["dim_u"],
            dim_x0=dim_x,
            dim_x1=dim_x,
            dim_y=dim_y,
            dim_z=model_params["dim_z"] + 1,
            dim_zm=model_params["dim_zm"],
            sig_u=sig_u,
            sig_x=model_params["sig_x"],
            sig_y=model_params["sig_y"],
            sig_z=sig_z,
            distributions=distributions,
            probit=False,
            heterox=False,
            heteroy=False,
            beta=beta,
            missing_x=model_params["missing_x"],
            detach_prop=False,
            sharedlatent=model_name == "our_snet_ident",
            c_index=c_index,
            prior_var=False,
            device="cpu",
        )
    elif model_name in ["our_sep", "our"]:
        model = General(
            f_z_rzx=arch_params["f_z_rzx"],
            f_u0_ru0=arch_params["f_u0_ru0"],
            f_u1_ru1=arch_params["f_u1_ru1"],
            f_ruz_x=arch_params["f_ruz_x"],
            f_z_y0=arch_params["f_z_y0"],
            f_z_y1=arch_params["f_z_y1"],
            f_z_t=arch_params["f_z_t"],
            g_x_z=arch_params["g_x_z"],
            g_x0_u0=arch_params["g_x0_u0"],
            g_x1_u1=arch_params["g_x1_u1"],
            g_m_zm=arch_params["g_m_zm"],
            f_zzmu_m=arch_params["f_zzmu_m"],
            dim_u0=model_params["dim_u"],
            dim_u1=model_params["dim_u"],
            dim_x0=dim_x,
            dim_x1=dim_x,
            dim_y=dim_y,
            dim_z=model_params["dim_z"],
            dim_zm=model_params["dim_zm"],
            sig_u=sig_u,
            sig_x=model_params["sig_x"],
            sig_y=model_params["sig_y"],
            sig_z=sig_z,
            reg_mmd=model_params["reg_mmd"],
            lam=model_params["lam"],
            discriminator=model_params["discr"],
            distributions=distributions,
            probit=False,
            heterox=False,
            heteroy=False,
            beta=beta,
            missing_x=model_params["missing_x"],
            detach_prop=False,
            sharedlatent=model_name == "our",
            device=device,
        )
    elif model_name in ["ident_sep", "ident"]:
        model = GeneralIdent(
            f_z_rzx=arch_params["f_z_rzx"],
            f_u0_ru0=arch_params["f_u0_ru0"],
            f_u1_ru1=arch_params["f_u1_ru1"],
            f_ruz_x=arch_params["f_ruz_x"],
            f_z_y0=arch_params["f_z_y0"],
            f_z_y1=arch_params["f_z_y1"],
            f_z_t=arch_params["f_z_t"],
            g_x_z=arch_params["g_x_z"],
            g_x0_u0=arch_params["g_x0_u0"],
            g_x1_u1=arch_params["g_x1_u1"],
            g_m_zm=arch_params["g_m_zm"],
            f_zzmu_m=arch_params["f_zzmu_m"],
            f_c_z=arch_params["f_c_z"],
            dim_u0=model_params["dim_u"],
            dim_u1=model_params["dim_u"],
            dim_x0=dim_x,
            dim_x1=dim_x,
            dim_y=dim_y,
            dim_z=model_params["dim_z"],
            dim_zm=model_params["dim_zm"],
            sig_u=sig_u,
            sig_x=model_params["sig_x"],
            sig_y=model_params["sig_y"],
            sig_z=sig_z,
            reg_mmd=model_params["reg_mmd"],
            lam=model_params["lam"],
            discriminator=model_params["discr"],
            distributions=distributions,
            probit=False,
            heterox=False,
            heteroy=False,
            beta=beta,
            missing_x=model_params["missing_x"],
            detach_prop=False,
            sharedlatent=model_name == "ident",
            c_index=c_index,
            prior_var=False,
            device="cpu",
        )
    elif model_name == "vae_y":
        model = NoUModel(
            f_z_x=arch_params["f_z_x"],
            f_z_y0=arch_params["f_z_y0"],
            f_z_y1=arch_params["f_z_y1"],
            f_z_t=arch_params["f_z_t"],
            g_x_z=arch_params["g_x_z"],
            dim_x0=dim_x,
            dim_x1=dim_x,
            dim_y=dim_y,
            dim_z=model_params["dim_z"],
            sig_x=model_params["sig_x"],
            sig_y=model_params["sig_y"],
            sig_z=sig_z,
            reg_mmd=model_params["reg_mmd"],
            lam=model_params["lam"],
            discriminator=model_params["discr"],
            distributions=distributions,
            probit=False,
            heterox=False,
            heteroy=False,
            beta=beta,
            detach_prop=False,
            device=device,
        )
    elif model_name == "vae":
        model = PlainVAE(
            f_z_x=arch_params["f_z_x"],
            g_x_z=arch_params["g_x_z"],
            dim_x0=dim_x,
            dim_x1=dim_x,
            dim_z=model_params["dim_z"],
            sig_x=model_params["sig_x"],
            sig_z=sig_z,
            distributions=distributions,
            beta=beta,
            device=device,
        )

    else:
        raise NotImplementedError(f"Model {model_name} is not implemented (yet)")

    return model


def train_model(
        data_dict,
        data_name,
        model,
        model_name,
        exp_name="tmp",
        n_epochs=100,
        lr=1e-3,
        wdecay=1e-4,
        grace_period=20,
        save_model=False,
):
    """
    Trains the model & data set combination with early stopping and saves the predictions
    """

    if not os.path.isdir(f"../runs/{data_name}/{exp_name}/{model_name}"):
        os.makedirs(f"../runs/{data_name}/{exp_name}/{model_name}")
    location = f"../runs/{data_name}/{exp_name}/{model_name}/"

    if th.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = model.to(device)
    model.device = device
    optim = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wdecay)
    log_losses = {"log_train": np.zeros(n_epochs), "log_val": np.zeros(n_epochs)}
    min_val = th.inf
    for epoch in tqdm(range(n_epochs), leave=False):
        for x, y, t, *_ in data_dict["train_loader"]:
            optim.zero_grad()

            if hasattr(model, "missing_x") and model.missing_x:
                x = x.to(device)
            else:
                x = x[:, : model.dim_x].to(device)
            y = y.to(device)
            t = t.to(device)

            if model_name in ["tarnet", "cfrnet", "singlenet", "tnet", "snet"]:
                loss = model.compute_loss(t=t, x=x, y=y)
            elif model_name in [
                "our_sep",
                "our",
                "vae_y",
                "vae",
                "ident_sep",
                "ident",
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
                loss = model.compute_loss(
                    t=t,
                    x=x,
                    y=y,
                    n_data=data_dict["N_dat"]["train"],
                )
            else:
                raise NotImplementedError(f"{model_name} is not implemented")
            if th.isnan(loss):
                break
            loss.backward()
            optim.step()

            log_losses["log_train"][epoch] += (
                    loss.item() / len(data_dict["train_loader"].dataset) * y.shape[0]
            )

        loss_val = validate(
            model, model_name, data_dict, data_dict["val_loader"], device=device
        )
        log_losses["log_val"][epoch] = loss_val
        if loss_val < min_val:
            th.save(model.state_dict(), location + "model.pt")
            min_val = loss_val
        if epoch > grace_period and all(
                loss_val > log_losses["log_val"][(epoch - grace_period): epoch]
        ):
            model.load_state_dict(th.load(location + "model.pt"))
            break

        if hasattr(model, "update_std") and model.update_std:
            if True:
                model.update_stdx(data_dict["train_loader"])
                model.update_stdy(data_dict["train_loader"])

    th.save(model.state_dict(), location + "model.pt")
    th.save(log_losses, location + "logs.pt")
    th.save(
        f"{data_name},{model_name},{exp_name},{n_epochs},{lr},{wdecay},{grace_period}",
        location + "hyper.pt",
    )

    if not save_model:
        os.remove(location + "model.pt")
        os.remove(location + "logs.pt")
        os.remove(location + "hyper.pt")

    # if data_name == "ihdp":
    #     save_predictions_ihdp(model, model_name, data_dict, location, device)


def validate(model, model_name, data_dict, loader, device="cpu"):
    """
    Compute the models loss wrt a specific loader
    """
    tot_loss = 0.0
    with th.no_grad():
        for x, y, t, *_ in loader:
            if hasattr(model, "missing_x") and model.missing_x:
                x = x.to(device)
            else:
                x = x[:, : model.dim_x].to(device)
            y = y.to(device)
            pop = t.to(device)
            if model_name in ["tarnet", "cfrnet", "singlenet", "tnet", "snet"]:
                loss = model.compute_loss(t=pop, x=x, y=y)
            elif model_name in [
                "our_sep",
                "our",
                "vae_y",
                "vae",
                "ident_sep",
                "ident",
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
                loss = model.compute_loss(
                    t=t,
                    x=x,
                    y=y,
                    n_data=data_dict["N_dat"]["val"],
                )
            tot_loss += loss.item() * y.shape[0] / len(loader.dataset)
    return tot_loss
