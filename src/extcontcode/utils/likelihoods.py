import torch as th
from torch import distributions as thd


def bernoulli_log_likelihood(log_pred, covar, mask, n_data, n_obs):
    return (
        (thd.Bernoulli(th.sigmoid(log_pred)).log_prob(covar) * mask).sum()
        * n_data
        / n_obs
    )


def normal_log_likelihood(pred_mean, pred_sigma, covar, mask, n_data, n_obs):
    return (
        (thd.Normal(pred_mean, pred_sigma).log_prob(covar) * mask).sum()
        * n_data
        / n_obs
    )


def cat_log_likelihood(log_pred, covar, n_data, n_obs):
    return (
        thd.Categorical(th.softmax(log_pred, 1)).log_prob(covar).sum() * n_data / n_obs
    )
