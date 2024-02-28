import math

import torch as th
import torch.distributions as thd
import torch.nn as nn

from extcontcode.models.tte.generaltimetoevent import GeneralTTE
from extcontcode.utils.utils import lvar_to_std


class GeneralIdentTTE(GeneralTTE):
    def __init__(
        self, f_c_z=nn.Sequential(nn.Identity()), c_index=(), prior_var=False, **kwargs
    ):
        super().__init__(**kwargs)

        self.lam = f_c_z

        self.c_index = c_index
        self.prior_var = prior_var

    def get_pz(self, c):
        tmp = self.lam(c)
        if self.prior_var:
            mu, lvar = tmp[:, : self.dim_z], tmp[:, self.dim_z :]
        else:
            mu = tmp
            lvar = 2 * math.log(self.sig_z) * th.ones_like(mu)
        sample = mu + lvar_to_std(lvar) * th.randn_like(mu)
        return sample, mu, lvar

    def get_kl_z(self, mean_z, lvar_z, n_train, xm):
        _, mean_prior, lvar_prior = self.get_pz(xm[:, self.c_index])

        return (
            thd.kl_divergence(
                thd.Normal(mean_z, th.exp(0.5 * lvar_z)),
                thd.Normal(mean_prior, lvar_to_std(lvar_prior)),
            ).mean()
            * n_train
        )
