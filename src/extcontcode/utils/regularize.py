import math

import numpy as np
import torch as th
import torch.distributions as thd
import torch.nn as nn


def kernelEQ(x, xp, scale=1.0, length=1.0):
    dist = (x[:, None] - xp[None]).pow(2).sum(2)
    return scale * th.exp(-0.5 * dist / length**2)


def kernelCat(x, xp):
    return 1.0 * (x == xp.T)


# See Murphy 2.7.3.3. for a variation that can help to
#   reduce everything to linear time if it turns out to be needed
def estimate_mmd(samples_p, samples_q, scale=1.0, length=1.0, kernel=kernelEQ):
    mmd = kernel(
        samples_p,
        samples_p,
        scale=scale,
        length=length,
    ).mean()
    mmd -= (
        2
        * kernel(
            samples_p,
            samples_q,
            scale=scale,
            length=length,
        ).mean()
    )
    mmd += kernel(
        samples_q,
        samples_q,
        scale=scale,
        length=length,
    ).mean()

    return mmd


#######
# JSD #
#######


def approx_jsd_normal(mean_p, lvar_p, mean_q, lvar_q, n_samples=10):
    "JSD = 0.5 (KL(P||M) + KL(Q||M), where M=0.5(P + Q)"
    P = thd.Normal(mean_p, th.exp(0.5 * lvar_p))
    Q = thd.Normal(mean_q, th.exp(0.5 * lvar_q))

    jsd = 0
    # Approximate KL(P||M)
    for _ in range(n_samples):
        sample = P.sample()
        jsd += (
            P.log_prob(sample).exp()
            - math.log(0.5)
            - th.log(P.log_prob(sample).exp() + Q.log_prob(sample).exp())
        )

    # Approximate KL(Q||M)
    for _ in range(n_samples):
        sample = Q.sample()
        jsd += (
            Q.log_prob(sample).exp()
            - math.log(0.5)
            - th.log(P.log_prob(sample).exp() + Q.log_prob(sample).exp())
        )
    jsd /= n_samples

    return jsd


def grad_reverse_fun(input, lam=1.0):
    return -lam * input + (1 + lam) * input.detach()


class GradientReversalLayer(nn.Module):
    """
    Class version of the `grad_reverse_fun` for more convenience in net definitions
    """

    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def forward(self, input):
        return grad_reverse_fun(input, self.lam)

    def extra_repr(self) -> str:
        return f"lam={self.lam}"


def grl_lambda_scheduler(p, scale_factor=1.0):
    # Note: gamma is a hyperparameter that was fixed in the original paper as 10
    gamma = 10.0
    return scale_factor * (2 / (1 + np.exp(-gamma * p)) - 1)


def fenchel_decomp(log_nu, samples_p, samples_q):
    """
    Estimate KL between q,p via the Fenchel dual form, i.e.
    KL(p,q) = max_nu (E_p[log_nu] - E_q[nu] + 1)
    """
    return log_nu(samples_p).mean() - log_nu(samples_q).exp().mean() + 1
