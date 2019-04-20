import unittest

import numpy as np

from hypothesis import given, settings
from hypothesis import strategies as st

from pyroc import ROC, bootstrap_roc
from pyroc.bootstrap import bootstrap_roc_

class TestBootstrap(unittest.TestCase):
    @settings(max_examples=1000)
    @given(gt=st.lists(st.booleans()), est=st.lists(st.floats()),
           seed=st.integers())
    def test_bootstrap_helper(self, gt, est, seed):
        if np.isnan(gt).any() or np.isnan(est).any() \
           or len(gt) != len(est) or len(gt) < 2:
            with self.assertRaises(ValueError):
                roc = ROC(gt, est)
                bootstrap_roc_((roc, seed))
        else:
            roc = ROC(gt, est)
            if seed < 0 or seed > 2**32 - 1:
                with self.assertRaises(ValueError):
                    bootstrap_roc_((roc, seed))
            else:
                bs_roc = bootstrap_roc_((roc, seed))
                assert np.isin(bs_roc.ground_truth, roc.ground_truth).all()
                assert np.isin(bs_roc.estimates, roc.estimates).all()

    def test_bootstrap_roc_ex1(self):
        gt = [True, True, False, False]
        est = [0.1, 0.3, 0.2, 0.4]
        roc = ROC(gt, est)

        result = bootstrap_roc(roc)
        assert len(result) == 1000

    def test_bootstrap_roc_ex2(self):
        ex_rng = np.random.RandomState(37)
        num = 10000
        ex_gt = ex_rng.binomial(1, 0.5, num)
        ex_est = ex_rng.rand((num))
        ex_roc = ROC(ex_gt, ex_est)
        ex_roc_auc_list = [roc.auc for roc in bootstrap_roc(ex_roc, seed=37)]

        assert np.isclose(np.mean(ex_roc_auc_list), 0.5042963196452369)
        assert np.isclose(np.var(ex_roc_auc_list)**.5, 0.006105232099260582)

    def test_bootstrap_roc_n_jobs(self):
        gt = [True, True, False, False]
        est = [0.1, 0.3, 0.2, 0.4]
        roc = ROC(gt, est)

        with self.assertRaises(RuntimeError):
            bootstrap_roc(roc, n_jobs=0)

        for n_jobs in [-2, -1, 1, 2, 4, 8, 16]:
            result = bootstrap_roc(roc, n_jobs=n_jobs)
            assert len(result) == 1000

    def test_bootstrap_roc_num_bootstraps(self):
        gt = [True, True, False, False]
        est = [0.1, 0.3, 0.2, 0.4]
        roc = ROC(gt, est)

        for num_bootstraps in [-1000, -1, 0]:
            with self.assertRaises(ValueError):
                bootstrap_roc(roc, num_bootstraps=num_bootstraps)

        for num_bootstraps in [1, 2, 8, 100, 1000, 10000]:
            result = bootstrap_roc(roc, num_bootstraps=num_bootstraps)
            assert len(result) == num_bootstraps
