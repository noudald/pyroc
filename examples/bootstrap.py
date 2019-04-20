"""Simple example to show how to use bootstrapping for ROC curves."""

import numpy as np

from pyroc import ROC, bootstrap_roc

# Simple example to test bootstrap
ex_rng = np.random.RandomState(37)
num = 10000
ex_gt = ex_rng.binomial(1, 0.5, num)
ex_est = ex_rng.rand((num))
ex_roc = ROC(ex_gt, ex_est)
ex_roc_auc_list = [roc.auc for roc in bootstrap_roc(ex_roc, seed=37)]

print(f'Average ROC AUC: {np.mean(ex_roc_auc_list)} +/- {np.var(ex_roc_auc_list)**.5}')
