import torch as th


def save_predictions_ihdp(model, model_name, data_dict, location, device="cpu"):
    train_preds = predict_ihdp(model, model_name, data_dict["train_loader"], device)
    val_preds = predict_ihdp(model, model_name, data_dict["val_loader"], device)
    test_preds = predict_ihdp(model, model_name, data_dict["test_loader"], device)
    th.save([train_preds, val_preds, test_preds], location + "preds.pt")


def predict_ihdp(model, model_name, loader, device="cpu"):
    """
    Compute and return the predictions of the model.
    NOTE: Currently this is IHDP specific in the loader structure
    """
    preds = []
    with th.no_grad():
        for (
            x,
            y,
            t,
            mu0,
            mu1,
        ) in loader:
            if hasattr(model, "missing_x") and model.missing_x:
                x = x.to(device)
            else:
                x = x[:, : model.dim_x].to(device)
            if model_name in [
                "our_sep",
                "our",
                "vae_y",
                "ident_sep",
                "ident",
                "tedvae",
                "our_snet",
                "our_snet_ident",
                "our_tedvae",
                "our_tedvae_ident",
                "our_snet_sep",
                "our_snet_ident_sep",
                "our_tedvae_sep",
                "our_tedvae_ident_sep",
            ]:
                pred = model.predict(x)
            elif model_name == "vae":
                pred = th.zeros_like(mu0)
            elif model_name in ["tarnet", "cfrnet", "singlenet", "tnet", "snet"]:
                pred = model.predict(x)

            if len(t.shape) == 1:
                t = t[:, None]
            joint = th.cat((mu0, mu1, t, pred.cpu()), 1)
            preds.append(joint)
    return th.cat(preds)
