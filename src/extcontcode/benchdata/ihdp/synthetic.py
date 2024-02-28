import numpy as np
import torch as th
from torch import distributions as thd


def create_synthetic_outcome_ihdp(
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

    M0 = th.exp((Xs[:, ind_pred] + 0.5).matmul(beta))
    M1 = Xs[:, ind_pred].matmul(beta)
    offset = M1[pop.eq(1)].mean() - M0[pop.eq(1)].mean() - 4
    M1 = M1 - offset

    Y0 = M0 + th.randn_like(M0)
    Y1 = M1 + th.randn_like(M1)
    Y = Y1
    Y[pop.eq(0)] = Y0[pop.eq(0)]

    Missing = th.ones((Xs.shape[0], 28))
    if missing:
        # If A == 1 the value is observed
        # The probability of being observed. If A==0 it is missing
        # p_disc_1_obs, p_disc_0_obs, p_cont = 0.2, 0.3, 0.1  # Strong
        # p_disc_1_obs, p_disc_0_obs, p_cont = 0.5, 0.7, 0.6  # Middle
        p_disc_1_obs, p_disc_0_obs, p_cont = 0.9, 0.9, 0.9  # Weak

        A = thd.Bernoulli(p_disc_1_obs * Xs[:, :19]).sample()
        A += thd.Bernoulli(p_disc_0_obs * (1 - Xs[:, :19])).sample()
        Missing[:, :19][A == 0] = 0
        Median = Xs[:, :19][Missing[:, :19] == 1].median(0)[0]
        Xs[:, :19][A == 0] = Median

        A = thd.Bernoulli(p_cont * (Xs[:, 19:24] > 0)).sample()
        Drop = A - 1.0 * (Xs[:, 19:24] > 0)
        Missing[:, 19:24][Drop == -1] = 0
        Xs[:, 19:24][Drop == -1] = 0

    return Xs, Y, M0, M1, beta, ids, shift, prob_flip, ind_pred, Missing
