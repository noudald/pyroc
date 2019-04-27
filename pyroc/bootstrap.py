"""Bootstrapping - for calculating the confidence interval for a ROC curve."""

import multiprocessing as mp

from typing import List, Optional, Tuple

import numpy as np

from pyroc import ROC


def bootstrap_roc_(args: Tuple[ROC, int]) -> ROC:
    """Helper method which applies bootstrapping on a given ROC curve.

    Parameters
    ----------
    args
        Tuple containing of an ROC object and a seed for applying
        bootstrapping.

    Returns
    -------
    bs_roc
        A bootstrapped ROC using the input ROC and the given seed.

    """
    cur_roc, seed = args
    bs_roc = cur_roc.bootstrap(seed)
    bs_roc.roc()
    return bs_roc


def bootstrap_roc(inp_roc: ROC,
                  num_bootstraps: int = 1000,
                  seed: Optional[int] = None,
                  n_jobs: int = -1) -> List[ROC]:
    """Bootstrap ROC curve using the DeLong method with concurrency support.

    Parameters
    ----------
    roc
        ROC curve object on which we apply the DeLong bootstrapping method.
    num_bootstraps
        Number of bootstraps to apply on the ROC curve. The number of ROC
        curves returned by this method is equal to num_bootstraps.
    seed
        Seed used for bootstrapping the ROC curve. If no seed is set the seed
        will be set randomly, generating non-deterministic output.
    n_jobs
        Number of jobs used to compute the bootstraps for the ROC curve in
        parallel. If n_jobs is set negative all available cpu threads will be
        used.

    Raises
    ------
    ValueError
        If num_bootstraps is not positive.
    RuntimeError
        If n_jobs is set to zero.

    Returns
    -------
    List of bootstrapped ROC curves.

    """
    if num_bootstraps < 1:
        raise ValueError('Non-positive bootstrap values are not allowed.')

    if n_jobs < 0:
        n_jobs = mp.cpu_count()
    elif n_jobs == 0:
        raise RuntimeError('Cannot initialize zero jobs.')

    rng = np.random.RandomState(seed)
    seeds = rng.randint(2**32 - 1, size=(num_bootstraps))

    with mp.Pool(processes=n_jobs) as pool:
        roc_list = pool.map(bootstrap_roc_, zip(len(seeds) * [inp_roc], seeds))

    return roc_list
