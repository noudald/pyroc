"""Simple example to show how to use bootstrapping for ROC curves."""

import matplotlib.pyplot as plt
import numpy as np

from pyroc import ROC, bootstrap_roc

# Simple example to test bootstrap
ex_rng = np.random.RandomState(37)
num = 100
ex_gt = ex_rng.binomial(1, 0.5, num)
ex_est = ex_rng.rand((num))
ex_roc = ROC(ex_gt, ex_est)
ex_roc_list = bootstrap_roc(ex_roc, seed=37)
ex_roc_auc_list = [roc.auc for roc in ex_roc_list]

print(f'Average ROC AUC: {np.mean(ex_roc_auc_list)}'
      f' +/- {np.var(ex_roc_auc_list)**.5}')

ax = ex_roc.plot(bootstrap=True,
                 num_bootstraps=1000,
                 seed=37,
                 num_bootstrap_jobs=-1,
                 color='red',
                 p_value=0.05,
                 mean_roc=False,
                 plot_roc_curve=True,
                 show_min_max=False)
ax = ex_roc.plot(bootstrap=True,
                 num_bootstraps=1000,
                 seed=37,
                 num_bootstrap_jobs=-1,
                 color='blue',
                 p_value=0.35,
                 mean_roc=True,
                 show_min_max=False,
                 ax=ax)

plt.show()
