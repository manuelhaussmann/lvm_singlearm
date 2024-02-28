# NOTE: For now we follow Franklin et al. (2014) who discuss a couple of metrics
# to measure covariate balance. Of the ten discussed metrics, they find that
# their proposal as well as the standardized absolute difference perform best.
# Since their proposal relies on propensity scores which we will for now avoid,
# we rely on the simpler standardized difference, i.e., metric (2) in their paper.
import torch as th
from sklearn.metrics import roc_auc_score


def diff(X1, X2, X1_mask=None, X2_mask=None, avg=True):
    """
    Assume dim(X1) and dim(X2) to be of n_samples x n_covariates
    """
    if X1_mask is None:
        X1_mask = th.ones_like(X1)
    if X2_mask is None:
        X2_mask = th.ones_like(X2)

    if avg:
        return th.mean(th.abs(mean_miss(X1, X1_mask) - mean_miss(X2, X2_mask)))
    else:
        return th.abs(mean_miss(X1, X1_mask) - mean_miss(X2, X2_mask))


def mean_miss(X1, mask=None):
    """
    Compute mean under missingness
    """
    if mask is None:
        mask = th.ones_like(X1)
    if len(X1.shape) == 1:
        N_obs = mask.sum()
        return (X1 * mask).sum() / N_obs
    else:
        N_obs = mask.sum(0)
        return (X1 * mask).sum(0) / N_obs


def cov_miss(X1, X2, X1_mask=None, X2_mask=None):
    """
    Compute covariance under missingness
    """
    if X1_mask is None:
        X1_mask = th.ones_like(X1)
    if X2_mask is None:
        X2_mask = th.ones_like(X1)
    return mean_miss(X1 * X2, X1_mask * X2_mask) - mean_miss(X1, X1_mask) * mean_miss(
        X2, X2_mask
    )


def var_miss(X1, mask=None):
    """
    Compute variance under missingness
    """
    if mask is None:
        mask = th.ones_like(X1)
    return mean_miss(X1.pow(2), mask) - mean_miss(X1, mask).pow(2)


def standardized_diff(X1, X2, X1_mask=None, X2_mask=None, verbose=False, full=False):
    """
    Compute the standardized difference (see metric (1) in Franklin et al., (2014))
    """
    D = diff(X1, X2, X1_mask, X2_mask, avg=False)
    s2x1 = var_miss(X1, X1_mask)
    s2x2 = var_miss(X2, X2_mask)

    if sum(s2x1.eq(0) * s2x2.eq(0)) > 0:
        if verbose:
            print(
                "WARNING: We are dropping some covariates due to zero overall variance, even though there is some difference in them"
            )
        if full:
            return D / th.sqrt((s2x1 + s2x2) / 2)

        else:
            tmp = D / th.sqrt((s2x1 + s2x2) / 2)
            tmp[th.isnan(tmp)] = 0  # in case of 0/0
            if sum(th.isinf(tmp)) > 0:
                print(
                    "WARNING: We have some fully distinct covariates, and this term is thus not standardized."
                )
            tmp[th.isinf(tmp)] = D[th.isinf(tmp)]
            return th.mean(tmp)

    else:
        if full:
            return D / th.sqrt((s2x1 + s2x2) / 2)
        else:
            return th.mean(D / th.sqrt((s2x1 + s2x2) / 2))


def post_matching_C(treat_assignment, prop_score):
    """
    Compute the post_matching statistic (see metric (9) in Franklin et al., (2014))
    """
    # returns [0.5, 1.0] where 0.5 is perfect confusion, and 1.0 is perfect prediction
    # We return it as score - 0.5 to make 0 the perfect balance
    return roc_auc_score(treat_assignment.numpy(), prop_score.numpy()) - 0.5


def gwd(X1, X2, X1_mask=None, X2_mask=None):
    """
    Compute the general weighted difference  (see metric (10) in Franklin et al., (2014))
    """
    X = th.cat((X1, X2))
    C = X1.shape[1]
    T = th.cat((th.zeros(X1.shape[0]), th.ones(X2.shape[0])))
    if X1_mask is None:
        X1_mask = th.ones_like(X1)
    if X2_mask is None:
        X2_mask = th.ones_like(X2)

    M = th.cat((X1_mask, X2_mask))

    tmp = th.zeros((X.shape[0], int(C * (C + 1) / 2)))
    tmp_miss = th.zeros((X.shape[0], int(C * (C + 1) / 2)))
    c = 0
    for i in range(C):
        for j in range(i, C):
            tmp[:, c] = X[:, i] * X[:, j]
            tmp_miss[:, c] = M[:, i] * M[:, j]
            c += 1

    tmp = th.cat((X, tmp), 1)
    tmp_miss = th.cat((M, tmp_miss), 1)
    D = diff(
        tmp[T.eq(0)], tmp[T.eq(1)], tmp_miss[T.eq(0)], tmp_miss[T.eq(1)], avg=False
    )
    V1 = var_miss(tmp[T.eq(0)], tmp_miss[T.eq(0)])
    V2 = var_miss(tmp[T.eq(1)], tmp_miss[T.eq(1)])
    W = th.cat((th.ones(C), 0.5 * th.ones(int(C * (C + 1) / 2)))) / th.sqrt(
        (V1 + V2) / 2
    )
    W[D.eq(0)] = 0
    res = W * D
    assert th.isnan(res).sum() == 0
    return res.sum() / (C + (C * (C + 1) / 2))


general_weighted_difference = lambda X1, X2, X1_mask=None, X2_mask=None: gwd(
    X1, X2, X1_mask, X2_mask
)
