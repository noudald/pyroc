"""Tools for comparing ROC curves with AUC."""

from typing import Optional, Tuple

import numpy as np

from scipy.stats import norm

from pyroc import bootstrap_roc, ROC

def compare_bootstrap(
        roc1: ROC,
        roc2: ROC,
        alt_hypothesis: float = 0.05,
        seed: Optional[int] = None) -> Tuple[bool, float]:
    """Compute roc1 < roc2 with alternative hypothesis using bootstrapping."""
    bootstrap_auc1 = np.array(list(roc.auc
                                   for roc in bootstrap_roc(roc1, seed=seed)))
    bootstrap_auc2 = np.array(list(roc.auc
                                   for roc in bootstrap_roc(roc2, seed=seed)))

    aucs = bootstrap_auc2 - bootstrap_auc1
    sample = np.mean(aucs) / np.std(aucs)

    p_value = 1 - norm.cdf(sample)

    if p_value < alt_hypothesis:
        return True, p_value
    return False, p_value
