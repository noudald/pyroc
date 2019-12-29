"""Tools for comparing ROC curves with AUC."""

from math import erf
from typing import Optional, Tuple

import numpy as np

from pyroc import bootstrap_roc, ROC


def gaussian_cdf(x: float) -> float:
    """Gaussian cummulative distribution function for N(0, 1).

    Parameters
    ----------
    x
        Quantile for which to compute the cummulative distribution.

    Returns
    -------
    Cummulative distribution for quantile x for Gaussian distribution N(0, 1).
    """
    return (1.0 + erf(x / 2.0**.5)) / 2.0

def compare_bootstrap(
        roc1: ROC,
        roc2: ROC,
        alt_hypothesis: float = 0.05,
        seed: Optional[int] = None) -> Tuple[bool, float]:
    """Compute roc1 < roc2 with alternative hypothesis using DeLong
    bootstrapping.

    The idea behind the this algorithm is to bootstrap roc1 and roc2, and
    compute the AUC (Area Under the Curve) for each of the bootstraps for roc1
    and roc2. For each bootstraps of roc1 and roc2 we compute the difference of
    the AUCs of ROC curves. Let

        aucs_diff = [auc11 - auc21, auc12 - auc22, ..., auc1n - auc2n],

    where auc1i is the AUC of ith bootstrap of roc1, and auc2i is the AUC of
    the ith bootstrap of roc2. We define a new stochast by

        Z = mean(aucs_diff) / std(aucs_diff).

    We assume that Z ~ N(0, 1), i.e. Z is drawn from a Gaussian distribution
    centered around 0 with standard deviation 1. Our zero hypothesis is that
    roc1 >= roc2, or in other words that P(Z) < 1 - alt_hypothesis. So that our
    alternative hypothesis is that roc1 < roc2. We reject the zero hypothesis
    if P(Z) > 1 - alt_hypothesis.

    Parameters
    ----------
    roc1
        The "to be assumed" smaller ROC curve than roc2.
    roc2
        The "to be assumed" larger ROC curve than roc1.
    alt_hypothesis
        The density for which we reject the zero hypothesis, and for which we
        therefore accept roc1 < roc2.
    seed
        Seed used for DeLong bootstrapping. If no seed is given a random seed
        will be used, resulting in non-deterministic results.

    Raises
    ------
    ValueError
        If alt_hypothesis is not between 0 and 1.

    Returns
    -------
    Tuple of a boolean and the p-value. I.e. the boolean represents if we can
    accept the alternative hypothesis roc1 < roc2, and the p-value represents
    the strength with which we accept the alternative hypothesis roc1 < roc2.

    """
    if not 0 <= alt_hypothesis <= 1:
        raise ValueError('Alternative hypothesis must be between 0 and 1.')

    bootstrap_auc1 = np.array(list(roc.auc
                                   for roc in bootstrap_roc(roc1, seed=seed)))
    bootstrap_auc2 = np.array(list(roc.auc
                                   for roc in bootstrap_roc(roc2, seed=seed)))

    aucs = bootstrap_auc2 - bootstrap_auc1
    sample = np.mean(aucs)
    if np.std(aucs) > 0:
        sample /= np.std(aucs)

    p_value = 1 - gaussian_cdf(sample)

    return p_value < alt_hypothesis, p_value

def compare_binary(
        roc1: ROC,
        roc2: ROC,
        alt_hypothesis: float = 0.05,
        seed: Optional[int] = None) -> Tuple[bool, float]:
    """Compute roc1 < roc2 using binary comparison with bootstrapping.

    The idea behind the this algorithm is to bootstrap roc1 and roc2, and
    compute the AUC (Area Under the Curve) for each of the bootstraps for roc1
    and roc2. For each bootstraps of roc1 and roc2 we compute the difference of
    the AUCs of ROC curves. Let

        aucs_diff = [auc11 - auc21, auc12 - auc22, ..., auc1n - auc2n],

    where auc1i is the AUC of ith bootstrap of roc1, and auc2i is the AUC of
    the ith bootstrap of roc2. We define the p-value, for which we can reject
    the zero hypothesis roc1 > roc2 as

        p_value = sum(aucs_diff > 0) / n.

    If p_value is smaller than alt_hypothesis we accept the alternative
    hypothesis roc1 < roc2.

    Parameters
    ----------
    roc1
        The "to be assumed" smaller ROC curve than roc2.
    roc2
        The "to be assumed" larger ROC curve than roc1.
    alt_hypothesis
        The density for which we reject the zero hypothesis, and for which we
        therefore accept roc1 < roc2.
    seed
        Seed used for DeLong bootstrapping. If no seed is given a random seed
        will be used, resulting in non-deterministic results.

    Raises
    ------
    ValueError
        If alt_hypothesis is not between 0 and 1.

    Returns
    -------
    Tuple of a boolean and the p-value. I.e. the boolean represents if we can
    accept the alternative hypothesis roc1 < roc2, and the p-value represents
    the strength with which we accept the alternative hypothesis roc1 < roc2.

    """
    if not 0 <= alt_hypothesis <= 1:
        raise ValueError('Alternative hypothesis must be between 0 and 1.')

    bootstrap_auc1 = np.array(list(roc.auc
                                   for roc in bootstrap_roc(roc1, seed=seed)))
    bootstrap_auc2 = np.array(list(roc.auc
                                   for roc in bootstrap_roc(roc2, seed=seed)))

    aucs = bootstrap_auc2 - bootstrap_auc1
    p_value = sum(aucs <= 0) / aucs.size

    return p_value < alt_hypothesis, p_value
