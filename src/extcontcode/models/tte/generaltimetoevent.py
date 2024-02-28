import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from extcontcode.models.generalsetup import GeneralModel
from extcontcode.utils.utils import softplus


class GeneralTTE(GeneralModel):
    def __init__(
        self,
        k=1,
        **kwargs,
    ):
        """
        The notation here is that the generator f_A_B maps from A to B
            while the inference net g_B_A maps B to A
        """
        super().__init__(**kwargs)

        self.k = k
        self.is_beta_param = True
        if self.is_beta_param:
            self.betacontr = nn.Parameter(th.randn(self.dim_z, 1))
            self.betatreat = nn.Parameter(th.randn(self.dim_z, 1))
        else:
            self.betacontr = nn.Sequential(
                nn.Linear(self.dim_z, 100),
                nn.ELU(),
                nn.Linear(100, 100),
                nn.ELU(),
                nn.Linear(100, 1),
            )
            self.betatreat = nn.Sequential(
                nn.Linear(self.dim_z, 100),
                nn.ELU(),
                nn.Linear(100, 100),
                nn.ELU(),
                nn.Linear(100, 1),
            )

    def get_py_tz(self, t, y, dmy, z):
        # y is the observed event time
        # my is a missingness indicator
        # delta is the censoring indicator
        my, delta = dmy[:, :1], dmy[:, 1:]

        if self.is_beta_param:
            lambdacontr = softplus(z.matmul(self.betacontr)) + 1e-8
            lambdatreat = softplus(z.matmul(self.betatreat)) + 1e-8
        else:
            lambdacontr = softplus(self.betacontr(F.elu(z))) + 1e-8
            lambdatreat = softplus(self.betatreat(F.elu(z))) + 1e-8

        log_term = lambda y, lam, k, delta: delta * (
            th.log(k / lam) + (k - 1) * th.log((y + 1e-8) / lam)
        ) - (y / lam).pow(k)

        t = t[:, None]
        assert t.shape == y.shape

        log_contr = log_term(y, lambdacontr, self.k, delta) * (y > 0)
        log_treat = log_term(y, lambdatreat, self.k, delta) * (y > 0)
        logp = ((1 - t) * log_contr + t * log_treat) * my
        return logp.sum() / sum(my)

    def predict(self, x, median=False):
        "Return the Weibull mean or median"
        _, z_mean, _ = self.get_qz_x(x)

        if self.is_beta_param:
            lambdacontr = softplus(z_mean.matmul(self.betacontr))
            lambdatreat = softplus(z_mean.matmul(self.betatreat))
        else:
            lambdacontr = softplus(self.betacontr(F.elu(z_mean))) + 1e-8
            lambdatreat = softplus(self.betatreat(F.elu(z_mean))) + 1e-8

        if median:
            y0 = lambdacontr * (math.log(2)) ** (1 / self.k)
            y1 = lambdatreat * (math.log(2)) ** (1 / self.k)
        else:
            y0 = lambdacontr * math.gamma(1 + 1 / self.k)
            y1 = lambdatreat * math.gamma(1 + 1 / self.k)

        return th.cat((y0, y1), 1)
