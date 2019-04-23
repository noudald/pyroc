"""Simple example how to compare two ROC curves."""

import numpy as np

from pyroc import ROC, compare_bootstrap

# Simple example to test bootstrap
ex_rng = np.random.RandomState(37)
num = 100
ex_gt = ex_rng.binomial(1, 0.5, num)
ex_est = ex_rng.rand((num))
ex_roc1 = ROC(ex_gt, ex_est)

ex_roc2 = ROC([True, True, True, False, False, False],
              [.9, .8, .35, .4, .3, .1])

print(compare_bootstrap(ex_roc1, ex_roc2))
print(ex_roc1 < ex_roc2)
print(ex_roc1 <= ex_roc2)
print(ex_roc1 > ex_roc2)
print(ex_roc1 >= ex_roc2)
print(ex_roc1 == ex_roc2)
