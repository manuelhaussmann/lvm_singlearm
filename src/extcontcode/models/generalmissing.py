import torch as th
import torch.distributions as thd
import torch.nn as nn

from extcontcode.models.generalsetup import GeneralModel
from extcontcode.utils.likelihoods import (
    bernoulli_log_likelihood,
    normal_log_likelihood,
    cat_log_likelihood,
)
from extcontcode.utils.regularize import estimate_mmd
from extcontcode.utils.utils import check_sorted_dists, lvar_to_std


class GeneralMissing(GeneralModel):
    """
    A generic models of our generative pipeline. It corresponds to the factorization p(t,u,x,y,z) = p(t|z)p(u)p(x|u,z)p(y|t,z)p(z)
    together with the variational posterior given by q(u,z) = q(u;x)q(z;x). (u is either shared or unique per population)
    Specific notes on the current setup:
        * MNAR is modeled as formulated in Collier et al. (2020)
        * It currently allows for homo- and heteroscedastic noise models (homoscedastic are learned via gd)
        * Currently we expect exactly one external and one treatment arm
        * We allow for two different populations which can have both treatment and control arms within them

    """

    def __init__(
        self,
        f_zzmu_m=nn.Sequential(nn.Identity()),
        g_m_zm=nn.Sequential(nn.Identity()),
        dim_zm=10,
        **kwargs,
    ):
        """
        The notation here is that the generator f_A_B maps from A to B
            while the inference net g_B_A maps B to A
        """
        super().__init__(**kwargs)

        # Take some defaults from the others
        self.sig_zm = self.sig_z
        self.dim_zm = dim_zm

        self.f_zzmu_m = f_zzmu_m
        self.g_m_zm = g_m_zm

        self.dim_x0 = int(self.dim_x0 / 2)
        self.dim_x1 = int(self.dim_x1 / 2)

        # Needed for the mask assignment in the elbo function
        assert self.dim_x0 == self.dim_x1, "Assume for now that they are equal"

    def get_pzm_samples(self, zm):
        sample = self.sig_zm * th.randn_like(zm)
        return sample, th.zeros_like(sample), self.sig_zm * th.ones_like(sample)

    def get_pm_uzzm(self, t, m, u0, u1, zm, z):
        mpred0 = self.f_zzmu_m(th.cat((u0, zm[t.eq(0)], z[t.eq(0)]), 1))
        mpred1 = self.f_zzmu_m(th.cat((u1, zm[t.eq(1)], z[t.eq(1)]), 1))
        res = (
            thd.Bernoulli(th.sigmoid(mpred0)).log_prob(m[t.eq(0)]).mean()
            + thd.Bernoulli(th.sigmoid(mpred1)).log_prob(m[t.eq(1)]).mean()
        )
        return res

    def get_qzm_m(self, m):
        out = self.g_m_zm(m)
        mean, lvar = out[:, : self.dim_zm], out[:, self.dim_zm :]

        lvar = self.clamp(lvar)
        sample_zm = mean + th.exp(0.5 * lvar) * th.randn_like(mean)
        return sample_zm, mean, lvar

    def estimate_elbo(self, t, xm, y, n_data, return_all=False, pop=None):
        """
        Compute an ELBO estimate
        :param t: binary treatment indicator
        :param xm: covariates + missing
        :param y: observed outcome
        :param n_data: list of observations
        :param return_all: return individual terms for debugging purposes
        :param pop: Preliminary population specific indicator (not yet properly iplemented)
        :return: the elbo/a list of elbo terms
        """
        assert pop is None, "For now pop and t are assumed to be identical"
        if len(t.shape) != 1:
            t = t.flatten()
        if pop is None:
            pop = t

        y, my = y[:, : self.dim_y], y[:, self.dim_y :]

        m = xm[:, self.dim_x0 :]
        xm0 = xm[t.eq(0)]
        xm1 = xm[t.eq(1)]
        if self.missing_x:
            x0, m0 = xm0[:, : self.dim_x0], xm0[:, self.dim_x0 :]
            x1, m1 = xm1[:, : self.dim_x1], xm1[:, self.dim_x1 :]

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

        if self.sharedlatent:
            sample_u, mean_u, lvar_u = self.get_qu_x(xm)
        else:
            sample_u0, mean_u0, lvar_u0 = self.get_qu0_x0(xm0)
            sample_u1, mean_u1, lvar_u1 = self.get_qu1_x1(xm1)

        sample_z, mean_z, lvar_z = self.get_qz_x(xm)

        sample_zm, mean_zm, lvar_zm = self.get_qzm_m(m)

        # Get log p(y|t,z)
        log_py_tz = self.get_py_tz(t, y, my, sample_z)

        data_fit = log_py_tz * n_train

        if self.sharedlatent:
            data_fit += (
                self.get_pm_uzzm(
                    t, m, sample_u[t.eq(0)], sample_u[t.eq(1)], sample_zm, sample_z
                )
                * n_train
            )
        else:
            data_fit += (
                self.get_pm_uzzm(t, m, sample_u0, sample_u1, sample_zm, sample_z)
                * n_train
            )

        # Get log p(x|u,z) (in log space for everything discrete and IR otherwise)
        if self.sharedlatent:
            pred_tx0, pred_tx1 = self.get_px_uz(pop, sample_u, sample_u, sample_z)
        else:
            pred_tx0, pred_tx1 = self.get_px_uz(pop, sample_u0, sample_u1, sample_z)

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
                        pred_tx0[:, n_ber : (n_ber + 2 * n_norm) : 2],
                        lvar_to_std(pred_tx0[:, (n_ber + 1) : (n_ber + 2 * n_norm) : 2])
                        + 1e-8,
                        x0[:, n_ber : (n_ber + n_norm)],
                        m0[:, n_ber : (n_ber + n_norm)],
                        n_cont,
                        sum(pop.eq(0)),
                    )
                else:
                    data_fit += normal_log_likelihood(
                        pred_tx0[:, n_ber : (n_ber + n_norm)],
                        self.sig_x0,
                        x0[:, n_ber : (n_ber + n_norm)],
                        m0[:, n_ber : (n_ber + n_norm)],
                        n_cont,
                        sum(pop.eq(0)),
                    )
            if sum(pop.eq(1)) > 0:
                if self.heterox:
                    data_fit += normal_log_likelihood(
                        pred_tx1[:, n_ber : (n_ber + 2 * n_norm) : 2],
                        lvar_to_std(pred_tx1[:, (n_ber + 1) : (n_ber + 2 * n_norm) : 2])
                        + 1e-8,
                        x1[:, n_ber : (n_ber + n_norm)],
                        m1[:, n_ber : (n_ber + n_norm)],
                        n_treat,
                        sum(pop.eq(1)),
                    )
                else:
                    data_fit += normal_log_likelihood(
                        pred_tx1[:, n_ber : (n_ber + n_norm)],
                        self.sig_x1,
                        x1[:, n_ber : (n_ber + n_norm)],
                        m1[:, n_ber : (n_ber + n_norm)],
                        n_treat,
                        sum(pop.eq(1)),
                    )
        if n_cat > 0:
            lcats_tx0 = x0[:, (n_ber + n_norm) :]
            lcats_tx1 = x1[:, (n_ber + n_norm) :]
            if self.heterox:
                lcats_pred_x0 = pred_tx0[:, (n_ber + 2 * n_norm) :]
                lcats_pred_x1 = pred_tx1[:, (n_ber + 2 * n_norm) :]
            else:
                lcats_pred_x0 = pred_tx0[:, (n_ber + n_norm) :]
                lcats_pred_x1 = pred_tx1[:, (n_ber + n_norm) :]
            i_count = 0
            for j, categorical in enumerate(
                [dist for dist in self.distributions if "Categorical" in dist]
            ):
                n_class = int(categorical.split("_")[1])
                if sum(pop.eq(0)) > 0:
                    data_fit += cat_log_likelihood(
                        lcats_pred_x0[:, i_count : (i_count + n_class)],
                        th.argmax(lcats_tx0[:, i_count : (i_count + n_class)], 1),
                        n_cont,
                        sum(pop.eq(0)),
                    )
                if sum(pop.eq(1)) > 0:
                    data_fit += cat_log_likelihood(
                        lcats_pred_x1[:, i_count : (i_count + n_class)],
                        th.argmax(lcats_tx1[:, i_count : (i_count + n_class)], 1),
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

        # Get the KL terms
        if self.sharedlatent:
            _, mean_prior, lvar_prior = self.get_pui(mean_u)
            kl_u = (
                thd.kl_divergence(
                    thd.Normal(mean_u, lvar_to_std(lvar_u)),
                    thd.Normal(mean_prior, lvar_to_std(lvar_prior)),
                ).mean()
                * n_train
            )
        if not self.sharedlatent:
            if sum(pop.eq(0)) > 0:
                _, mean_prior, lvar_prior = self.get_pui(mean_u0)
                kl_u0 = (
                    thd.kl_divergence(
                        thd.Normal(mean_u0, lvar_to_std(lvar_u0)),
                        thd.Normal(mean_prior, lvar_to_std(lvar_prior)),
                    ).mean()
                    * n_cont
                )
            else:
                kl_u0 = 0.0
            if sum(pop.eq(1)) > 0:
                _, mean_prior, lvar_prior = self.get_pui(mean_u1)
                kl_u1 = (
                    thd.kl_divergence(
                        thd.Normal(mean_u1, lvar_to_std(lvar_u1)),
                        thd.Normal(mean_prior, lvar_to_std(lvar_prior)),
                    ).mean()
                    * n_treat
                )
            else:
                kl_u1 = 0.0
            kl_u = kl_u0 + kl_u1

        kl_zm = (
            thd.kl_divergence(
                thd.Normal(mean_zm, th.exp(0.5 * lvar_zm)),
                thd.Normal(th.zeros_like(mean_zm), self.sig_z * th.ones_like(lvar_zm)),
            ).mean()
            * n_train
        )

        kl_z = (
            thd.kl_divergence(
                thd.Normal(mean_z, th.exp(0.5 * lvar_z)),
                thd.Normal(th.zeros_like(mean_z), self.sig_z * th.ones_like(lvar_z)),
            ).mean()
            * n_train
        )

        if self.reg_mmd > 0 and sum(t.eq(0)) > 0 and sum(t.eq(1)) > 0:
            mmd = estimate_mmd(mean_z[t.eq(0)], mean_z[t.eq(1)])
        else:
            mmd = 0.0

        if return_all:
            return (
                (data_fit + log_pt_z - kl_u - kl_zm - self.beta * kl_z) / n_train
                - self.reg_mmd * mmd,
                data_fit / n_train,
                log_py_tz / n_train,
                log_pt_z / n_train,
                kl_u0 / n_train,
                kl_u1 / n_train,
                kl_zm / n_train,
                kl_z / n_train,
                mmd,
            )
        else:
            return (
                data_fit + log_pt_z - kl_u - kl_zm - self.beta * kl_z
            ) / n_train - self.reg_mmd * mmd
