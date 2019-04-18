"""Bootstrapping - for calculating the confidence interval for a ROC curve."""

import multiprocessing as mp

from typing import List, Optional, Tuple

import numpy as np

from pyroc import ROC


def bootstrap_roc_(args: Tuple[ROC, int]) -> ROC:
    """Helper for bootstrapping ROC curve."""
    cur_roc, seed = args
    bs_roc = cur_roc.bootstrap(seed)
    bs_roc.roc()
    return bs_roc


def bootstrap_roc(inp_roc: ROC,
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
        roc_list = pool.map(bootstrap_roc_, zip(len(seeds) * [inp_roc], seeds))

    return roc_list


if __name__ == '__main__':
    # Simple example to test bootstrap
    rng = np.random.RandomState(37)
    num = 10000
    gt = rng.binomial(1, 0.5, num)
    est = rng.rand((num))
    roc = ROC(gt, est)
    roc_list = bootstrap_roc(roc, seed=37)
