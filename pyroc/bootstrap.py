"""Bootstrapping - for calculating the confidence interval for a ROC curve."""

import multiprocessing as mp

from typing import List, Optional, Tuple

import numpy as np

from pyroc import ROC


def bootstrap_roc_(args: Tuple[ROC, int]):
    """Helper for bootstrapping ROC curve."""
    roc, seed = args
    bootstrap_roc = roc.bootstrap(seed)
    bootstrap_roc.roc()
    return bootstrap_roc


def bootstrap_roc(roc: ROC,
                  num_bootstraps: int = 1000,
                  seed: Optional[int] = None,
                  n_jobs: int = -1) -> List[ROC]:
    """Bootstrap ROC curve.

    Args:
        roc: ROC cruve object to bootstrap.
        num_bootstraps: Number of bootstraps to apply.
        seed: Random seed for selecting bootstraps.
        n_jobs: Number of jobs to use for computing bootstraps. If n_jobs is
            set to -1 all available cpu threads will be used.

    Returns:
        List of bootstrapped ROC curves.

    """
    if n_jobs < 0:
        n_jobs = mp.cpu_count()

    rng = np.random.RandomState(seed)
    seeds = rng.randint(2**32 - 1, size=(num_bootstraps))

    with mp.Pool(processes=n_jobs) as pool:
        roc_list = pool.map(bootstrap_roc_, zip(len(seeds) * [roc], seeds))

    return roc_list


if __name__ == '__main__':
    # Simple example to test bootstrap
    rng = np.random.RandomState(37)
    num = 10000
    gt = rng.binomial(1, 0.5, num)
    est = rng.rand((num))
    roc = ROC(gt, est)
    roc_list = bootstrap_roc(roc, seed=37)
