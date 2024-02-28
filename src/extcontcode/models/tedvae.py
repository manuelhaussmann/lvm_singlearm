import math

import torch as th
import torch.distributions as thd
import torch.nn as nn
import torch.nn.functional as F

from extcontcode.utils.likelihoods import (
    bernoulli_log_likelihood,
    normal_log_likelihood,
    cat_log_likelihood,
)
from extcontcode.utils.regularize import GradientReversalLayer, estimate_mmd
from extcontcode.utils.utils import check_sorted_dists, lvar_to_std


class TEDVAE(nn.Module):
    def __init__(
            self,
            f_z_x=nn.Sequential(nn.Identity()),
            f_z_y0=nn.Sequential(nn.Identity()),
            f_z_y1=nn.Sequential(nn.Identity()),
            f_z_t=nn.Sequential(nn.Identity()),
            g_x_zt=nn.Sequential(nn.Identity()),
            g_x_zc=nn.Sequential(nn.Identity()),
            g_x_zy=nn.Sequential(nn.Identity()),
            dim_x0=1,
            dim_x1=1,
            dim_y=1,
            dim_z=1,
            sig_x=1.0,
            sig_y=1.0,
            sig_z=1.0,
            reg_mmd=0.0,
            lam=0.0,
            discriminator=True,
            distributions=None,
            probit=False,
            heterox=False,
            heteroy=False,
            beta=1.0,  # Allows for the beta-VAE setting
            detach_prop=False,
            missing_x=False,
            device="cpu",
    ):
        """
        The notation here is that the generator f_A_B maps from A to B
            while the inference net g_B_A maps B to A
        """
        super().__init__()

        # Neural network-based mapping
        ##   f_a_b the generative mapping a -> b (rX gives us a preliminary representation)
        self.f_z_x = f_z_x
        self.f_z_y0 = f_z_y0
        self.f_z_y1 = f_z_y1
        # Discriminator formulation following Ganin and Lempitsky (2015)
        if discriminator:
            self.f_z_t = nn.Sequential(GradientReversalLayer(lam=lam), *f_z_t)
            reg_mmd = 0
        else:
            self.f_z_t = f_z_t

        ## g_a_b the inferential mapping a -> b (rX gives us a preliminary representation)
        self.g_x_zt = g_x_zt
        self.g_x_zc = g_x_zc
        self.g_x_zy = g_x_zy

        # stds of the different continuous distributions
        # allowing for homoscedastic as well as heteroscedastic observation noise

        self.sig_x0 = nn.Parameter(
            sig_x * th.ones(sum("Normal" == X for X in distributions))
        )
        self.sig_x1 = nn.Parameter(
            sig_x * th.ones(sum("Normal" == X for X in distributions))
        )

        self.sig_y = nn.Parameter(sig_y * th.ones(1))
        self.sig_z = sig_z

        self.heterox = heterox
        self.heteroy = heteroy
        self.missing_x = missing_x

        # dimensionalities of
        self.dim_x0 = dim_x0
        self.dim_x1 = dim_x1
        if self.dim_x0 == self.dim_x1:
            self.dim_x = self.dim_x0
        self.dim_y = dim_y
        self.dim_z = dim_z
        if self.dim_x0 == self.dim_x1:
            self.dim_x = self.dim_x0

        assert (
                self.dim_x0 == self.dim_x1
        ), "Sorry, for the current distributions implementation we need them to be the same"

        # Regularizer for the individual terms
        self.reg_mmd = reg_mmd  # Regularize the MMD
        self.beta = beta  # switch from VAE to beta-VAE style scaling of the KL
        self.lam = lam
        self.discriminator = discriminator

        # May you protect our variances from numerical problems
        self.is_clamp = True
        self.clamp = lambda x: x.clamp(-40, 20) if self.is_clamp else x
        self.detach_prop = detach_prop
        self.probit = probit
        if probit:
            assert len(self.f_z_t) == 1 and isinstance(
                self.f_z_t[0], nn.Linear
            ), "Using probit for p(t|z). Remember that discriminator needs to be linear in that case"

        # A list of the shape [N_tot, N_con, N_val] for each of train, val, test
        if distributions is None:
            self.distributions = ["Normal"] * self.dim_x0
        else:
            self.distributions = distributions

        self.device = device
        self.debug = False  # Turn on manually if you have to

    def get_pz(self, inp):
        sample = self.sig_z * th.randn_like(inp)
        return (
            sample,
            th.zeros_like(sample),
            2 * math.log(self.sig_z) * th.ones_like(sample),
        )

    def get_pui(self, inp):
        sample = self.sig_u * th.randn_like(inp)
        return (
            sample,
            th.zeros_like(sample),
            2 * math.log(self.sig_u) * th.ones_like(sample),
        )

    def get_px_z(self, pop, z):
        "Get the latent input and split it into the two domains after a joint representation"
        repr_z = self.f_z_x(z)

        return repr_z[pop.eq(0)], repr_z[pop.eq(1)]

    def pred_px_uz(self, pop, z, props=False):
        x0, x1 = self.get_px_z(pop, z)

        i = 0
        for dist in self.distributions:
            if dist == "Bernoulli":
                if props:
                    x0[:, i] = th.sigmoid(x0[:, i])
                    x1[:, i] = th.sigmoid(x1[:, i])
                else:
                    x0[:, i] = x0[:, i] > 0
                    x1[:, i] = x1[:, i] > 0
                i += 1
            elif dist == "Normal":
                i += 1
            elif "Categorical" in dist:
                n_class = int(dist.split("_")[1])
                if props:
                    x0[:, i] = th.softmax(x0[:, i: (i + n_class)], 1)
                    x1[:, i] = th.softmax(x1[:, i: (i + n_class)], 1)
                    i += n_class
                else:
                    x0[:, i] = th.argmax(x0[:, i: (i + n_class)], 1)
                    x1[:, i] = th.argmax(x1[:, i: (i + n_class)], 1)
                    x0 = th.cat((x0[:, : (i + 1)], x0[:, (i + n_class):]), 1)
                    x1 = th.cat((x1[:, : (i + 1)], x1[:, (i + n_class):]), 1)
                    i += 1
            else:
                raise NotImplementedError(f"Sorry, {dist} is not implemented")
        return x0, x1

    def get_py_tz(self, t, y, my, z):
        """
        Get the latent input and split it into the two domains after a joint representation"
        """

        z = F.elu(z)

        assert self.dim_z == 5
        zt, zc, zy = z[:, :1], z[:, 1:3], z[:, 3:]

        y0 = self.f_z_y0(th.cat((zc, zy), 1))
        y1 = self.f_z_y1(th.cat((zc, zy), 1))

        assert y0.shape == y.shape
        assert t[:, None].shape == y.shape

        if self.heteroy:
            res = (
                          thd.Normal(y0[:, :1], y0[:, 1:]).log_prob(y) * (1.0 - t[:, None]) * my
                  ).mean() + (
                          thd.Normal(y1[:, :1], y1[:, 1:]).log_prob(y) * t[:, None] * my
                  ).mean()
        else:
            res = (
                          thd.Normal(y0, self.sig_y).log_prob(y) * (1.0 - t[:, None]) * my
                  ).mean() + (thd.Normal(y1, self.sig_y).log_prob(y) * t[:, None] * my).mean()

        return res

    def predict(self, x):
        _, z_mean, _ = self.get_qz_x(x)
        z = F.elu(z_mean)
        assert self.dim_z == 5
        zt, zc, zy = z[:, :1], z[:, 1:3], z[:, 3:]

        y0 = self.f_z_y0(th.cat((zc, zy), 1))
        y1 = self.f_z_y1(th.cat((zc, zy), 1))
        return th.cat((y0, y1), 1)

    def get_pt_z(self, t, z):
        """
        Allows for a probit scaled likelihood which can be computed analytically  Normal input
        """

        assert self.dim_z == 5
        z = F.elu(z[:, :3])

        out = self.f_z_t(z.detach() if self.detach_prop else z)
        res = thd.Bernoulli(th.sigmoid(out)).log_prob(1.0 * t[:, None]).mean()
        return th.sigmoid(out), res

    def get_z(self, x):
        assert self.dim_z == 5

        zt = self.g_x_zt(x)
        zc = self.g_x_zc(x)
        zy = self.g_x_zy(x)

        m_z = th.cat((zt[:, :1], zc[:, :2], zy[:, :2]), 1)
        lvar_z = th.cat((zt[:, 1:], zc[:, 2:], zy[:, 2:]), 1)

        return m_z, lvar_z

    def get_qz_x(self, xm):
        mean, lvar = self.get_z(xm)

        lvar = self.clamp(lvar)
        sample_z = mean + lvar_to_std(lvar) * th.randn_like(mean)
        return sample_z, mean, lvar

    def get_kl_z(self, mean_z, lvar_z, n_train, placeholder):
        _, mean_prior, lvar_prior = self.get_pz(mean_z)
        return (
                thd.kl_divergence(
                    thd.Normal(mean_z, th.exp(0.5 * lvar_z)),
                    thd.Normal(mean_prior, lvar_to_std(lvar_prior)),
                ).mean()
                * n_train
        )

    def estimate_elbo(self, t, xm, y, n_data, return_all=False, pop=None):
        """
        Compute an ELBO estimate
        :param t: binary treatment indicator
        :param xm: covariates + missing
        :param y: observed outcome
        :param n_data: list of observations
        :param return_all: return individual terms for debugging purposes
        :param pop: Preliminary population specific indicator (not yet properly implemented)
        :return: the elbo/a list of elbo terms
        """
        assert pop is None, "For now pop and t are assumed to be identical"
        if len(t.shape) != 1:
            t = t.flatten()
        if pop is None:
            pop = t

        y, my = y[:, : self.dim_y], y[:, self.dim_y:]

        xm0 = xm[t.eq(0)]
        xm1 = xm[t.eq(1)]
        if self.missing_x:
            x0, m0 = xm0[:, : self.dim_x0], xm0[:, self.dim_x0:]
            x1, m1 = xm1[:, : self.dim_x1], xm1[:, self.dim_x1:]
        else:
            x0 = xm0
            x1 = xm1
            m0 = th.ones_like(xm0)
            m1 = th.ones_like(xm1)

        assert check_sorted_dists(
            self.distributions
        ), "Sorry, the distributions need to be sorted"
        n_ber = sum(["Bernoulli" in dist for dist in self.distributions])
        n_norm = sum(["Normal" in dist for dist in self.distributions])
        n_cat = sum(["Categorical" in dist for dist in self.distributions])
        n_train = n_data[0]
        n_cont = n_data[1]
        n_treat = n_data[2]

        # NOTE: Missingness cannot be within the categorical variable
        if self.missing_x:
            assert n_cat == 0, "so far this is not properly implemented"

        sample_z, mean_z, lvar_z = self.get_qz_x(xm)

        # Get log p(y|t,z)
        log_py_tz = self.get_py_tz(t, y, my, sample_z)

        data_fit = log_py_tz * n_train

        # Get log p(x|u,z) (in log space for everything discrete and IR otherwise)
        pred_tx0, pred_tx1 = self.get_px_z(pop, sample_z)

        if n_ber > 0:
            if sum(pop.eq(0)) > 0:
                data_fit += bernoulli_log_likelihood(
                    pred_tx0[:, :n_ber],
                    x0[:, :n_ber],
                    m0[:, :n_ber],
                    n_cont,
                    sum(pop.eq(0)),
                )
            if sum(pop.eq(1)) > 0:
                data_fit += bernoulli_log_likelihood(
                    pred_tx1[:, :n_ber],
                    x1[:, :n_ber],
                    m1[:, :n_ber],
                    n_treat,
                    sum(pop.eq(1)),
                )
        if n_norm > 0:
            if sum(pop.eq(0)) > 0:
                if self.heterox:
                    data_fit += normal_log_likelihood(
                        pred_tx0[:, n_ber: (n_ber + 2 * n_norm): 2],
                        lvar_to_std(pred_tx0[:, (n_ber + 1): (n_ber + 2 * n_norm): 2])
                        + 1e-8,
                        x0[:, n_ber: (n_ber + n_norm)],
                        m0[:, n_ber: (n_ber + n_norm)],
                        n_cont,
                        sum(pop.eq(0)),
                    )
                else:
                    data_fit += normal_log_likelihood(
                        pred_tx0[:, n_ber: (n_ber + n_norm)],
                        self.sig_x0,
                        x0[:, n_ber: (n_ber + n_norm)],
                        m0[:, n_ber: (n_ber + n_norm)],
                        n_cont,
                        sum(pop.eq(0)),
                    )
            if sum(pop.eq(1)) > 0:
                if self.heterox:
                    data_fit += normal_log_likelihood(
                        pred_tx1[:, n_ber: (n_ber + 2 * n_norm): 2],
                        lvar_to_std(pred_tx1[:, (n_ber + 1): (n_ber + 2 * n_norm): 2])
                        + 1e-8,
                        x1[:, n_ber: (n_ber + n_norm)],
                        m1[:, n_ber: (n_ber + n_norm)],
                        n_treat,
                        sum(pop.eq(1)),
                    )
                else:
                    data_fit += normal_log_likelihood(
                        pred_tx1[:, n_ber: (n_ber + n_norm)],
                        self.sig_x1,
                        x1[:, n_ber: (n_ber + n_norm)],
                        m1[:, n_ber: (n_ber + n_norm)],
                        n_treat,
                        sum(pop.eq(1)),
                    )
        if n_cat > 0:
            lcats_tx0 = x0[:, (n_ber + n_norm):]
            lcats_tx1 = x1[:, (n_ber + n_norm):]
            if self.heterox:
                lcats_pred_x0 = pred_tx0[:, (n_ber + 2 * n_norm):]
                lcats_pred_x1 = pred_tx1[:, (n_ber + 2 * n_norm):]
            else:
                lcats_pred_x0 = pred_tx0[:, (n_ber + n_norm):]
                lcats_pred_x1 = pred_tx1[:, (n_ber + n_norm):]
            i_count = 0
            for j, categorical in enumerate(
                    [dist for dist in self.distributions if "Categorical" in dist]
            ):
                n_class = int(categorical.split("_")[1])
                if sum(pop.eq(0)) > 0:
                    data_fit += cat_log_likelihood(
                        lcats_pred_x0[:, i_count: (i_count + n_class)],
                        th.argmax(lcats_tx0[:, i_count: (i_count + n_class)], 1),
                        n_cont,
                        sum(pop.eq(0)),
                    )
                if sum(pop.eq(1)) > 0:
                    data_fit += cat_log_likelihood(
                        lcats_pred_x1[:, i_count: (i_count + n_class)],
                        th.argmax(lcats_tx1[:, i_count: (i_count + n_class)], 1),
                        n_treat,
                        sum(pop.eq(1)),
                    )
                i_count += n_class

        # Note: If we have a discriminator this implicitly turns it into a minimizer
        if self.probit:
            _, log_pt_z = self.get_pt_z(t, (mean_z, lvar_z.exp()))
        else:
            _, log_pt_z = self.get_pt_z(t, sample_z)
        log_pt_z *= n_train

        kl_z = self.get_kl_z(mean_z, lvar_z, n_train, xm)

        if self.reg_mmd > 0 and sum(t.eq(0)) > 0 and sum(t.eq(1)) > 0:
            mmd = estimate_mmd(mean_z[t.eq(0)], mean_z[t.eq(1)])
        else:
            mmd = 0.0

        if return_all:
            return (
                (data_fit + log_pt_z - self.beta * kl_z) / n_train - self.reg_mmd * mmd,
                data_fit / n_train,
                log_py_tz / n_train,
                log_pt_z / n_train,
                kl_z / n_train,
                mmd,
            )
        else:
            return (
                    data_fit + log_pt_z - self.beta * kl_z
            ) / n_train - self.reg_mmd * mmd

    def compute_loss(self, t, x, y, n_data, return_all=False, pop=None):
        """
        Generic loss function
        :param t: binary treatment indicator
        :param x: covariates
        :param y: observed outcome
        :param n_data: list of observations
        :param return_all: return individual terms for debugging purposes
        :param pop: Preliminary population specific indicator (not yet properly iplemented)
        :return: the elbo/a list of elbo terms
        """
        if return_all:
            loss = self.estimate_elbo(t, x, y, n_data, return_all, pop)
            return -loss[0], *loss[1:]
        else:
            return -self.estimate_elbo(t, x, y, n_data, return_all, pop)
