import click
import yaml
from tqdm import tqdm

from extcontcode.benchdata.ihdp.ihdp import load_ihdp_setting
from extcontcode.utils.experiments import generate_model, train_model


def load_exp_data(data_set, exp_setting, params, perc_val=0.1):
    """
    Load data for a specific replication ID
    """
    assert data_set in [
        "ihdp",
    ], f"The experiment {data_set} is not implemented"
    assert exp_setting in [
        "full",
        "partial",
        "tte",
    ], f"The setup {exp_setting} is not available"

    partial = exp_setting == "partial"
    tte = exp_setting == "tte"

    if data_set == "ihdp":
        id = params["id"]
        # train_loader, val_loader, test_loader, mean_y, std_y, mean_x, std_x, distributions, N_data,
        return load_ihdp_setting(
            id,
            perc_val,
            n_pred=params["npred"],
            shift_size=params["shift"],
            prob_flip=params["prob_flip"],
            partial=partial,
            tte=tte,
        )


@click.command()
@click.option("--exp_setup", default="")
@click.option("--min_id", default=-1)
@click.option("--max_id", default=-1)
@click.option("--verbose", default=False, is_flag=True)
def run_exp(
    exp_setup,
    min_id,
    max_id,
    verbose,
):
    """
    Run the experiment specified in the exp_setting
    :param exp_setup: expects a yaml file or dictionary with the parameters
    :param min_id: specifies the min id of the range of iterations to train on
    :param max_id: specifies the max id of the range of iterations to train on
    """
    with open(exp_setup, "r") as f:
        exp_setup = yaml.safe_load(f)

    dataset = exp_setup["dataset"]
    exp_setting = exp_setup["exp_setting"]
    data_params = exp_setup["data_params"]
    model_params = exp_setup["model_params"]

    assert dataset == "ihdp", f"Dataset: {dataset} not available"
    assert exp_setting in [
        "full",
        "partial",
        "tte",
    ], f"Exp-Setting: {exp_setting} not available"

    if dataset == "ihdp":
        if min_id != -1:
            data_params["idstart"] = min_id
        if min_id != -1:
            data_params["id"] = max_id
        if data_params["id"] == "all":
            ids = list(range(1000))
        else:
            ids = list(range(data_params["idstart"], data_params["id"]))
    else:
        raise NotImplementedError(f"{dataset} is not implemented")

    shifts = data_params["shift"]
    flips = data_params["prob_flip"]
    for model_name in exp_setup["model_name"]:
        assert model_name in [
            "our_snet",
            "our_snet_ident",
            "our_snet_sep",
            "our_snet_ident_sep",
            "snet",
            "singlenet",
            "tnet",
            "tarnet",
            "cfrnet",
            "our_sep",
            "our",
            "vae_y",
            "vae",
            "ident_sep",
            "ident",
            "tedvae",
            "our_tedvae",
            "our_tedvae_ident",
            "our_tedvae_sep",
            "our_tedvae_ident_sep",
        ], f"Model: {model_name} not available"

        if verbose:
            print(f"INFO: Running {dataset}, {exp_setting}, {model_name}")

        for id in tqdm(ids, leave=False):
            for shift in shifts:
                data_params["shift"] = shift
                for flip in flips:
                    data_params["prob_flip"] = flip

                    data_dict = load_exp_data(
                        dataset,
                        exp_setting,
                        {
                            "id": id,
                            "npred": data_params["npred"],
                            "shift": data_params["shift"],
                            "prob_flip": data_params["prob_flip"],
                        },
                    )
                    model = generate_model(
                        model_name,
                        model_params,
                        data_dict["distributions"],
                        c_index=data_dict["c_index"],
                        tte=exp_setting == "tte",
                    )

                    train_model(
                        data_dict,
                        dataset,
                        model,
                        model_name,
                        exp_name=exp_setup["exp_name"]
                        + f"_z{model_params['dim_z']}"
                        + f"_npred{data_params['npred']}_shift{data_params['shift']}_flip{data_params['prob_flip']}"
                        + f"_{id}",
                        n_epochs=exp_setup["exp_params"]["n_epochs"],
                        lr=float(exp_setup["exp_params"]["lr"]),
                        wdecay=float(exp_setup["exp_params"]["wdecay"]),
                        grace_period=exp_setup["exp_params"]["grace_period"],
                        save_model=exp_setup["exp_params"]["save_model"],
                    )


if __name__ == "__main__":
    run_exp()
