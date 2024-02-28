import torch as th
from torch import nn as nn
from torch.nn import functional as F

from extcontcode.utils.regularize import estimate_mmd
from extcontcode.utils.utils import softplus


class TARNetTTE(nn.Module):
    """
    Implementation of the TARNet (Shalit et al. 2017).
    """

    def __init__(
        self,
        z_nn=nn.Sequential(),
        y0_nn=nn.Sequential(),
        y1_nn=nn.Sequential(),
        dim_x=1,
        dim_z=1,
        k=1,
    ):
        super().__init__()
        self.z_nn = z_nn
        self.y0_nn = y0_nn
        self.y1_nn = y1_nn
        self.dim_x = dim_x
        self.dim_z = dim_z
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

    def compute_loss(self, t, x, y):
        z = self.z_nn(x)
        return -self.tte_loss(z, t, y)

    def tte_loss(self, z, t, y_true):
        # y is the observed event time
        # my is a missingness indicator
        # delta is the censoring indicator

        y, dmy = y_true[:, :1], y_true[:, 1:]
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

        t = t
        assert t.shape == y.shape

        log_contr = log_term(y, lambdacontr, self.k, delta) * (y > 0)
        log_treat = log_term(y, lambdatreat, self.k, delta) * (y > 0)
        logp = ((1 - t) * log_contr + t * log_treat) * my
        return logp.sum() / sum(my)


class CFRNetTTE(TARNetTTE):
    def __init__(self, reg_mmd=1, **kwargs):
        super().__init__(**kwargs)
        self.reg_mmd = reg_mmd

    def compute_loss(self, t, x, y):
        z = self.z_nn(x)
        loss_factual = self.tte_loss(z, t, y)
        t = t.squeeze()
        if t.eq(0).sum() > 0 and t.eq(1).sum() > 0:
            mmd = estimate_mmd(z[t.eq(0)], z[t.eq(1)])
        else:
            mmd = 0
        loss = loss_factual + self.reg_mmd * mmd
        return loss


class SNetTTE(nn.Module):
    def __init__(
        self,
        z0=nn.Sequential(),
        z01=nn.Sequential(),
        z1=nn.Sequential(),
        za=nn.Sequential(),
        zt=nn.Sequential(),
        mu0=nn.Sequential(),
        mu1=nn.Sequential(),
        pi=nn.Sequential(),
        dim_x=1,
        dim_z=1,
        k=1,
    ):
        super().__init__()
        self.z0 = z0
        self.z01 = z01
        self.z1 = z1
        self.za = za
        self.zt = zt
        self.m0 = mu0
        self.m1 = mu1
        self.pi = pi
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.k = k
        self.is_beta_param = True
        if self.is_beta_param:
            self.betacontr = nn.Parameter(th.randn(4, 1))
            self.betatreat = nn.Parameter(th.randn(4, 1))
        else:
            self.betacontr = nn.Sequential(
                nn.Linear(4, 100),
                nn.ELU(),
                nn.Linear(100, 100),
                nn.ELU(),
                nn.Linear(100, 1),
            )
            self.betatreat = nn.Sequential(
                nn.Linear(4, 100),
                nn.ELU(),
                nn.Linear(100, 100),
                nn.ELU(),
                nn.Linear(100, 1),
            )

    def ortho_reg(self):
        def _get_absolute_rowsums(W):
            return th.sum(W.abs(), dim=0)

        w_z0 = _get_absolute_rowsums(self.z0[0].weight)
        w_z1 = _get_absolute_rowsums(self.z1[0].weight)
        w_z01 = _get_absolute_rowsums(self.z01[0].weight)
        w_za = _get_absolute_rowsums(self.za[0].weight)
        w_zt = _get_absolute_rowsums(self.zt[0].weight)

        return th.sum(
            w_z0 * (w_z1 + w_z01 + w_za + w_zt)
            + w_z1 * (w_z01 + w_za + w_zt)
            + w_z01 * (w_za + w_zt)
            + w_za * w_zt
        )

    def forward(self, x):
        z0 = F.elu(self.z0(x))
        z01 = F.elu(self.z01(x))
        z1 = F.elu(self.z1(x))
        za = F.elu(self.za(x))
        zt = F.elu(self.zt(x))

        y0 = self.m0(th.cat((z0, z01, za), 1))
        y1 = self.m1(th.cat((z1, z01, za), 1))
        pi = self.pi(th.cat((za, zt), 1))

        return th.cat([y0, y1, pi], dim=1)

    def get_z(self, x):
        z0 = self.z0(x)
        z01 = self.z01(x)
        z1 = self.z1(x)
        za = self.za(x)
        zt = self.zt(x)
        return th.cat((z0, z01, z1, za, zt), 1)

    @th.no_grad()
    def predict(self, x):
        """Returns a tensor of predicted mu0 and mu1"""
        concat_pred = self.forward(x)
        return concat_pred[:, :-1]  # Don't predict the propensity score

    def compute_loss(self, t, x, y):
        """Computes the overall loss of the models."""
        loss = self.tte_loss(x, t, y)
        return loss

    def tte_loss(self, x, t, y_true):
        # y is the observed event time
        # my is a missingness indicator
        # delta is the censoring indicator

        y, dmy = y_true[:, :1], y_true[:, 1:]
        my, delta = dmy[:, :1], dmy[:, 1:]

        z0 = F.elu(self.z0(x))
        z01 = F.elu(self.z01(x))
        z1 = F.elu(self.z1(x))
        za = F.elu(self.za(x))
        zt = F.elu(self.zt(x))

        z0 = th.cat((z0, z01, za), 1)
        z1 = th.cat((z1, z01, za), 1)
        tpred = self.pi(th.cat((za, zt), 1))
        assert tpred.shape == t.shape
        assert t.shape == my.shape

        if self.is_beta_param:
            lambdacontr = softplus(z0.matmul(self.betacontr)) + 1e-8
            lambdatreat = softplus(z1.matmul(self.betatreat)) + 1e-8
        else:
            lambdacontr = softplus(self.betacontr(F.elu(z0))) + 1e-8
            lambdatreat = softplus(self.betatreat(F.elu(z1))) + 1e-8

        log_term = lambda y, lam, k, delta: delta * (
            th.log(k / lam) + (k - 1) * th.log((y + 1e-8) / lam)
        ) - (y / lam).pow(k)

        t = t
        assert t.shape == y.shape

        log_contr = log_term(y, lambdacontr, self.k, delta) * (y > 0)
        log_treat = log_term(y, lambdatreat, self.k, delta) * (y > 0)
        logp = ((1 - t) * log_contr + t * log_treat) * my
        loss = logp.sum() / sum(my)
        loss += th.binary_cross_entropy_with_logits(tpred, target=t).mean()
        loss += self.ortho_reg()
        return loss
