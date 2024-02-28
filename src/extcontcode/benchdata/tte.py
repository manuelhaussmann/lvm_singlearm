import numpy as np
import torch as th
import torch.distributions as thd

from extcontcode.utils.utils import softplus


def gen_survival_data(X, T, beta=None, T0=365, pcens=0.3):
    if beta is None:
        beta = th.randn(X.shape[1])

    outcome = gen_survival_outcome(X, beta, T0, pcens)

    t_obs = th.zeros(X.shape[0])
    delta_obs = th.zeros(X.shape[0])

    t_obs[T.eq(0)] = outcome[0][0][T.eq(0)]
    t_obs[T.eq(1)] = outcome[1][0][T.eq(1)]

    delta_obs[T.eq(0)] = outcome[0][1][T.eq(0)]
    delta_obs[T.eq(1)] = outcome[1][1][T.eq(1)]

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
    # Very simplistic relationship
    rbase = th.clip(softplus(X.matmul(beta)), 0.5)
    rtreat = th.clip(softplus(X.matmul(beta)).sqrt() - 0.3, 0.5)

    t_base, delta_base = sample_based_on_risk(rbase, T0, pcens)
    t_treat, delta_treat = sample_based_on_risk(rtreat, T0, pcens)

    return (t_base, delta_base), (t_treat, delta_treat)


N = 60_000
X = th.randn(N, 3)
beta = th.randn(X.shape[1])
T = 1.0 * (th.randn(N) > 0)
X[T.eq(0)] += 3 + th.randn_like(X)[T.eq(0)]

obs_outcome, full_outcome = gen_survival_data(X, T)

if True:
    np.savez("tmp.npz", t=obs_outcome[0], clust=T, delta=obs_outcome[1])
    np.savez("foo.npz", all=full_outcome)
