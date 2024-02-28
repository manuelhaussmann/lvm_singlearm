import torch as th

from extcontcode.utils.balancemetrics import (
    standardized_diff,
    general_weighted_difference,
)


def get_matches(dist, T, max_dist=False):
    "Returns the id -1 for a matching that is too large"
    # collects the matched ids
    res_id = th.zeros_like(T).long()
    # collects the matched distances
    res_dist = th.zeros(T.shape, dtype=dist.dtype)
    # Get the submatrix of shape T0 x T1 and get the minimum over the rows
    #   as the matches for the treated population
    val, id = dist[T.eq(0)][:, T.eq(1)].min(0)
    # Generate a list of ids; pick the subset of T0, and pick the ids of these
    res_id[T.eq(1)] = th.arange(len(T))[T.eq(0)][id]
    res_dist[T.eq(1)] = val
    # Repeat the procedure for the control population
    val, id = dist[T.eq(0)][:, T.eq(1)].min(1)
    res_id[T.eq(0)] = th.arange(len(T))[T.eq(1)][id]
    res_dist[T.eq(0)] = val
    if max_dist:
        res_id[res_dist > max_dist] = -1
    return res_id


def get_caliper_thresh(ps_scores, pop_index, scaling=0.2):
    "Following the suggestions of Austin (2011)"
    sqstds = 0.0
    logit_scores = th.logit(ps_scores)
    for i in pop_index.unique():
        sqstds += logit_scores[pop_index.eq(i)].std().pow(2)
    return scaling * th.sqrt(sqstds / len(pop_index.unique()))


def get_dist_thresh(dist, pop_index, use_log=True, scaling=0.2):
    "Adaption of the PS threshold, just without a proper theoretical foundation"
    if use_log:
        return scaling * dist[pop_index.eq(0)][:, pop_index.eq(1)].min(0)[0].log().std()
    else:
        return scaling * dist[pop_index.eq(0)][:, pop_index.eq(1)].min(0)[0].std()


def get_subset_matched(data, pop, matching):
    "Return the matched data frame with `-1` matches, i.e., failed ones, removed."

    return data[pop][matching[pop] != -1], data[matching[pop]][matching[pop] != -1]


def compute_matched_metrics(data, pop, mask=None, verbose=False):
    if mask is None:
        stddiff = standardized_diff(data[pop.eq(0)], data[pop.eq(1)])
    else:
        stddiff = standardized_diff(
            data[pop.eq(0)], data[pop.eq(1)], mask[pop.eq(0)], mask[pop.eq(1)]
        )

    if sum(pop.eq(0)) == sum(pop.eq(1)):
        gendiff = general_weighted_difference(
            data[pop.eq(0)], data[pop.eq(1)], mask[pop.eq(0)], mask[pop.eq(1)]
        )
    else:
        gendiff = th.nan

    if verbose:
        print("###")
        print(f"SD: {stddiff:.4f} // GWD: {gendiff:.4f}")

    return stddiff, gendiff


def comp_dist_prop(
    dist,
    ps,
    covar,
    pop,
    mask=None,
    max_dist=False,
    verbose=True,
    nogwd=False,
    full=False,
):
    distmatrix = dist(ps, ps)
    res = get_matches(distmatrix, pop, max_dist=max_dist)
    match_data = get_subset_matched(covar, pop.eq(1), res)
    if mask is not None:
        match_mask = get_subset_matched(mask, pop.eq(1), res)
    else:
        match_mask = (None, None)
    sd = standardized_diff(*match_data, *match_mask, full=full)
    if nogwd:
        gwd = 0.0
    else:
        gwd = general_weighted_difference(*match_data, *match_mask, full=full)

    if verbose:
        print("###")
        print(f"SD: {sd:.4f} // GWD: {gwd:.4f}")
    return sd, gwd


def comp_dist(
    dist,
    covar,
    match_covar,
    pop,
    mask=None,
    lvar=None,
    max_dist=False,
    verbose=True,
    nogwd=False,
    full=False,
):
    if lvar is None:
        distmatrix = dist(match_covar, match_covar)
    else:
        distmatrix = dist(
            match_covar, th.exp(0.5 * lvar), match_covar, th.exp(0.5 * lvar)
        )
    res = get_matches(distmatrix, pop, max_dist=max_dist)
    match_data = get_subset_matched(covar, pop.eq(1), res)
    if mask is not None:
        match_mask = get_subset_matched(mask, pop.eq(1), res)
    else:
        match_mask = (None, None)
    sd = standardized_diff(*match_data, *match_mask, full=full)
    if nogwd:
        gwd = 0.0
    else:
        gwd = general_weighted_difference(*match_data, *match_mask, full=full)
    if verbose:
        print("###")
        print(f"SD: {sd:.4f} // GWD: {gwd:.4f}")
    return sd, gwd
