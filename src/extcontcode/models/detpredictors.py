import torch as th
import torch.nn as nn
import torch.nn.functional as F

from extcontcode.utils.regularize import estimate_mmd


class TARNet(nn.Module):
    """
    Implementation of the TARNet (Shalit et al. 2017).
    """

    def __init__(
        self,
        z_nn=nn.Sequential(),
        y0_nn=nn.Sequential(),
        y1_nn=nn.Sequential(),
        dim_x=1,
    ):
        super().__init__()
        self.z_nn = z_nn
        self.y0_nn = y0_nn
        self.y1_nn = y1_nn
        self.dim_x = dim_x

    def forward(self, x):
        z = self.z_nn(x)
        z_elu = F.elu(z)
        y0 = self.y0_nn(z_elu)
        y1 = self.y1_nn(z_elu)
        return th.cat([y0, y1], dim=1)

    @th.no_grad()
    def predict(self, x):
        """Returns a tensor of predicted mu0 and mu1"""
        concat_pred = self.forward(x)
        return concat_pred

    def compute_loss(self, t, x, y):
        """Computes the overall loss of the models."""
        concat_pred = self.forward(x)
        loss = self.regression_loss(concat_pred, t, y)
        return loss

    def regression_loss(self, concat_pred, t_true, y_true):
        assert y_true.shape[1] == 2, "Need a scalar for the outcome + a mask"
        y_true, ym = y_true[:, :1], y_true[:, 1:]

        y0_pred = concat_pred[:, [0]]
        y1_pred = concat_pred[:, [1]]

        assert y_true.shape == y0_pred.shape
        assert y_true.shape == ym.shape
        assert y_true.shape == t_true.shape

        # The second factor in each removes the hidden output
        loss0 = th.mean((1.0 - t_true) * th.square(y_true - y0_pred) * ym)
        loss1 = th.mean(t_true * th.square(y_true - y1_pred) * ym)
        loss = loss0 + loss1
        return loss


class CFRNet(TARNet):
    """
    CFRNet= TARNet + MMD (Shalit et al., 2017)
    """

    def __init__(self, reg_mmd=1.0, **kwargs):
        super().__init__(**kwargs)
        self.reg_mmd = reg_mmd

    def forward(self, x):
        z = self.z_nn(x)
        z_elu = F.elu(z)
        y0 = self.y0_nn(z_elu)
        y1 = self.y1_nn(z_elu)
        return th.cat([y0, y1, z], dim=1)

    def compute_loss(self, t, x, y):
        concat_pred = self.forward(x)
        z = concat_pred[:, 2:]
        loss_factual = self.regression_loss(concat_pred, t, y)
        t = t.squeeze()
        if t.eq(0).sum() > 0 and t.eq(1).sum() > 0:
            mmd = estimate_mmd(z[t.eq(0)], z[t.eq(1)])
        else:
            mmd = 0
        loss = loss_factual + self.reg_mmd * mmd
        return loss


class SingleNet(nn.Module):
    def __init__(
        self,
        y_nn=nn.Sequential(),
        dim_x=1,
    ):
        super().__init__()
        self.y_nn = y_nn
        self.dim_x = dim_x

    def forward(self, x):
        y0 = self.y_nn(th.cat((x, th.zeros((x.shape[0], 1))), 1))
        y1 = self.y_nn(th.cat((x, th.ones((x.shape[0], 1))), 1))
        return th.cat([y0, y1], dim=1)

    @th.no_grad()
    def predict(self, x):
        """Returns a tensor of predicted mu0 and mu1"""
        concat_pred = self.forward(x)
        return concat_pred

    def compute_loss(self, t, x, y):
        """Computes the overall loss of the models."""
        concat_pred = self.forward(x)
        loss = self.regression_loss(concat_pred, t, y)
        return loss

    def regression_loss(self, concat_pred, t_true, y_true):
        assert y_true.shape[1] == 2, "Need a scalar for the outcome + a mask"
        y_true, ym = y_true[:, :1], y_true[:, 1:]

        y0_pred = concat_pred[:, [0]]
        y1_pred = concat_pred[:, [1]]

        assert y_true.shape == y0_pred.shape
        assert y_true.shape == ym.shape
        assert y_true.shape == t_true.shape

        # The second factor in each removes the hidden output
        loss0 = th.mean((1.0 - t_true) * th.square(y_true - y0_pred) * ym)
        loss1 = th.mean(t_true * th.square(y_true - y1_pred) * ym)
        loss = loss0 + loss1
        return loss


class TNet(nn.Module):
    def __init__(
        self,
        y0_nn=nn.Sequential(),
        y1_nn=nn.Sequential(),
        dim_x=1,
    ):
        super().__init__()
        self.y0_nn = y0_nn
        self.y1_nn = y1_nn
        self.dim_x = dim_x

    def forward(self, x):
        y0 = self.y0_nn(x)
        y1 = self.y1_nn(x)
        return th.cat([y0, y1], dim=1)

    @th.no_grad()
    def predict(self, x):
        """Returns a tensor of predicted mu0 and mu1"""
        concat_pred = self.forward(x)
        return concat_pred

    def compute_loss(self, t, x, y):
        """Computes the overall loss of the models."""
        concat_pred = self.forward(x)
        loss = self.regression_loss(concat_pred, t, y)
        return loss

    def regression_loss(self, concat_pred, t_true, y_true):
        assert y_true.shape[1] == 2, "Need a scalar for the outcome + a mask"
        y_true, ym = y_true[:, :1], y_true[:, 1:]

        y0_pred = concat_pred[:, [0]]
        y1_pred = concat_pred[:, [1]]

        assert y_true.shape == y0_pred.shape
        assert y_true.shape == ym.shape
        assert y_true.shape == t_true.shape

        # The second factor in each removes the hidden output
        loss0 = th.mean((1.0 - t_true) * th.square(y_true - y0_pred) * ym)
        loss1 = th.mean(t_true * th.square(y_true - y1_pred) * ym)
        loss = loss0 + loss1
        return loss


class SNet(nn.Module):
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
        concat_pred = self.forward(x)
        loss = self.regression_loss(concat_pred, t, y)
        return loss

    def regression_loss(self, concat_pred, t_true, y_true):
        assert y_true.shape[1] == 2, "Need a scalar for the outcome + a mask"
        y_true, ym = y_true[:, :1], y_true[:, 1:]

        y0_pred = concat_pred[:, [0]]
        y1_pred = concat_pred[:, [1]]
        tpred = concat_pred[:, [2]]

        assert y_true.shape == y0_pred.shape
        assert y_true.shape == ym.shape
        assert y_true.shape == t_true.shape
        assert tpred.shape == t_true.shape

        # The second factor in each removes the hidden output
        loss0 = th.mean((1.0 - t_true) * th.square(y_true - y0_pred) * ym)
        loss1 = th.mean(t_true * th.square(y_true - y1_pred) * ym)
        loss = loss0 + loss1
        loss += th.binary_cross_entropy_with_logits(tpred, target=t_true).mean()
        loss += self.ortho_reg()
        return loss
