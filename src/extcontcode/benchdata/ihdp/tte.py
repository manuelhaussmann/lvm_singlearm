import numpy as np
import torch as th
import torch.distributions as thd

from extcontcode.utils.utils import softplus


def gen_survival_data(X, T, beta=None, T0=365, pcens=0.3):
    if beta is None:
        beta = th.randn(X.shape[1])

    outcome = gen_survival_outcome(X, beta, T0, pcens)

    t_obs = th.zeros(X.shape[0], 1)
    delta_obs = th.zeros(X.shape[0], 1)

    t_obs[T.eq(0)] = outcome[0][0][T.eq(0)].float()
    t_obs[T.eq(1)] = outcome[1][0][T.eq(1)].float()

    delta_obs[T.eq(0)] = outcome[0][1][T.eq(0)].float()
    delta_obs[T.eq(1)] = outcome[1][1][T.eq(1)].float()

    return (t_obs, delta_obs), outcome


def sample_based_on_risk(r, T0, pcens):
    lam = th.exp(r) / T0
    a = th.rand_like(r)

    u = -th.log(a) / lam

    q = np.quantile(u, 1 - pcens)

    tcens = thd.Uniform(th.min(u), q).sample()

    delta = u <= tcens
    t = th.where(delta, u, tcens)

    return t, 1.0 * delta


def gen_survival_outcome(X, beta, T0=365, pcens=0.3):
    # Src. Roughly follows Manduchi et al. (2022) Sec E.2, and Pölsterl (2019)

    # Compute a separate risk for both groups
    # Very simple relationship for now
    rbase = th.clip(
        softplus(X.matmul(beta)) + (X - 0.5).matmul(beta).pow(2), min=0.5, max=15
    )
    rtreat = th.clip(softplus(X.matmul(beta) + X.matmul(beta).abs()), min=0.5, max=15)

    t_base, delta_base = sample_based_on_risk(rbase, T0, pcens)
    t_treat, delta_treat = sample_based_on_risk(rtreat, T0, pcens)

    return (t_base, delta_base), (t_treat, delta_treat)


def create_synthetic_outcome_ihdp_tte(
    X,
    pop,
    beta=None,
    ids=None,
    shift=None,
    missing=False,
    shift_size=0,
    prob_flip=0.0,
    ind_pred=0,
    sigma_shift=3.0,
    n_pred=20,
):
    """
    Create outcomes for a modified version of IHDP (fewer predictive, higher divergence) that retains the behaviour of the original
    """
    D_cov = X.shape[1]
    assert D_cov == 25, "Sorry I am specifically designed for IHDP"

    Xs = 1.0 * X

    # Weights for the shared covariates
    if beta is None:
        beta = th.from_numpy(
            np.random.choice(
                [0, 0.1, 0.2, 0.3, 0.4], size=(n_pred, 1), p=[0.6, 0.1, 0.1, 0.1, 0.1]
            )
        )
        ind_pred = np.random.choice(D_cov, n_pred, replace=False)
        ids = np.random.choice(19, 5, replace=False)
        shift = (thd.Bernoulli(0.5).sample((1, 5)) * 2 - 1) * shift_size
    else:
        beta = beta
        ind_pred = ind_pred
        ids = ids
        shift = shift
        prob_flip = prob_flip

    # Pick a random set of binary numbers with a bias in a certain direction
    if prob_flip > 0.0:
        label = thd.Bernoulli(prob_flip).sample((sum(pop.eq(0)), 5))
        # Flip two of these
        label[:, 2] = th.abs(label[:, 2] - 1)
        label[:, 4] = th.abs(label[:, 4] - 1)

        # Pick five random discrete variables and shift them around

        # Logical Or: If False + False -> stay false, else get/stay positive -> bias towards one
        # Logical And: If True + True -> stay true, else get/stay negative -> bias towards zero
        Xs[pop.eq(0), ids[0]] = th.logical_or(
            Xs[pop.eq(0), ids[0]], label[:, 0]
        ).double()
        Xs[pop.eq(0), ids[1]] = th.logical_or(
            Xs[pop.eq(0), ids[1]], label[:, 1]
        ).double()
        Xs[pop.eq(0), ids[2]] = th.logical_and(
            Xs[pop.eq(0), ids[2]], label[:, 2]
        ).double()
        Xs[pop.eq(0), ids[3]] = th.logical_or(
            Xs[pop.eq(0), ids[3]], label[:, 3]
        ).double()
        Xs[pop.eq(0), ids[4]] = th.logical_and(
            Xs[pop.eq(0), ids[4]], label[:, 4]
        ).double()

    if shift_size > 0:
        Xs[pop.eq(0), 19:24] += shift + sigma_shift * th.randn_like(
            Xs[pop.eq(0), 19:24]
        )

    obs_outcome, full_outcome = gen_survival_data(Xs[:, ind_pred], pop, beta=beta)

    Y = obs_outcome[0]
    delta = obs_outcome[1]

    M0 = th.stack(full_outcome[0]).transpose(0, 1).squeeze()
    M1 = th.stack(full_outcome[1]).transpose(0, 1).squeeze()

    Missing = th.ones((Xs.shape[0], 28))
    if missing:
        raise NotImplementedError

    return Xs, Y, M0, M1, beta, ids, shift, prob_flip, ind_pred, Missing, delta
