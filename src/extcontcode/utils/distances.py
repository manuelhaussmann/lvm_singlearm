import torch as th
from scipy.stats import wasserstein_distance


def compute_wasserstein_1d(X, Z):
    assert (
        X.shape[1] == Z.shape[1]
    ), f"Require the same dimensionality, but got {X.shape[1]} and {Z.shape[1]}"
    wasser = 0.0
    for i in range(X.shape[1]):
        wasser += wasserstein_distance(X[:, i], Z[:, i])
    return wasser


def compute_sqhellinger_diagnormal(mu1, sigma1, mu2, sigma2):
    """
    Hellinger between to multivariate Normal with  diagonal covariate matrices,
    via https://en.wikipedia.org/wiki/Hellinger_distance
    """
    detS1 = sigma1.prod(1).sqrt()
    detS2 = sigma2.prod(1).sqrt()
    detS = th.sqrt((0.5 * (sigma1.pow(2)[:, None] + sigma2.pow(2)[None])).prod(2))

    exponent = (
        -1
        / 4
        * (
            (mu1[:, None] - mu2[None]).pow(2)
            / (sigma1.pow(2)[:, None] + sigma2.pow(2)[None])
        ).sum(2)
    )

    return 1 - (detS1[:, None] * detS2[None]) / detS * th.exp(exponent)


def compute_sqhellinger_diagnormal_ind(mu1, sigma1, mu2, sigma2):
    """
    Hellinger between to multivariate Normal with  diagonal covariate matrices,
    via https://en.wikipedia.org/wiki/Hellinger_distance
    """
    idetS1 = th.prod(sigma1).sqrt()
    idetS2 = th.prod(sigma2).sqrt()
    idetS = th.prod(0.5 * (sigma1.pow(2) + sigma2.pow(2))).sqrt()

    iexponent = -1 / 4 * th.sum((mu1 - mu2).pow(2) / (sigma1.pow(2) + sigma2.pow(2)))

    return 1 - idetS1 * idetS2 / idetS * th.exp(iexponent)


# Vectorized Euclid helperfunction
def sq_euclid(x, z):
    return (x[:, None] - z[None]).pow(2).sum(2)


def absolute(x, z):
    return (x[:, None] - z[None]).abs().sum(2)


def compute_2wasserstein_diagnormal(mu1, sigma1, mu2, sigma2):
    """
    Computer the 2-Wasserstein distance between two multivariate Normal distributions,
    with diagonal covariate matrices via https://en.wikipedia.org/wiki/Wasserstein_metric
    """
    return sq_euclid(mu1, mu2) + sq_euclid(sigma1, sigma2)


def compute_2wasserstein_diagnormal_ind(mu1, sigma1, mu2, sigma2):
    """
    Computer the 2-Wasserstein distance between two multivariate Normal distributions,
    with diagonal covariate matrices via https://en.wikipedia.org/wiki/Wasserstein_metric
    """
    return (mu1 - mu2).pow(2).sum() + (sigma1 - sigma2).pow(2).sum()
