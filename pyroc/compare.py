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
    """Compute roc1 < roc2 with alternative hypothesis using bootstrapping."""
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
    """Compute roc1 < roc2 with binary comparison using bootstrapping."""
    if not 0 <= alt_hypothesis <= 1:
        raise ValueError('Alternative hypothesis must be between 0 and 1.')

    bootstrap_auc1 = np.array(list(roc.auc
                                   for roc in bootstrap_roc(roc1, seed=seed)))
    bootstrap_auc2 = np.array(list(roc.auc
                                   for roc in bootstrap_roc(roc2, seed=seed)))

    aucs = bootstrap_auc2 - bootstrap_auc1
    p_value = sum(aucs <= 0) / aucs.size

    return p_value < alt_hypothesis, p_value
