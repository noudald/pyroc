import unittest

import numpy as np

from hypothesis import given
from hypothesis import strategies as st

from pyroc import ROC


class TestROCExample1():
    def setup(self):
        self.gt = [0, 0, 1, 1]
        self.est = [0.1, 0.3, 0.2, 0.4]
        self.roc = ROC(self.gt, self.est)

    def test_roc(self):
        fps, tps, thresholds = self.roc.roc()

        np.testing.assert_allclose(fps, np.array([0, .5, 1]))
        np.testing.assert_allclose(tps, np.array([.5, .5, 1]))
        np.testing.assert_allclose(thresholds, np.array([.4, .3, .1]))

    def test_auc(self):
        auc = self.roc.auc

        np.testing.assert_allclose(5/8, auc)

class TestROCExample2():
    def test_auc_all_gt_equal(self):
        roc = ROC([0, 0, 0], [0, 0, 0])
        assert np.isclose(roc.auc, 1.0)

        roc = ROC([1, 1, 1], [1, 1, 1])
        assert np.isclose(roc.auc, 1.0)

    def test_auc_two_values(self):
        roc = ROC([0, 1], [0, 1])
        assert np.isclose(roc.auc, 1.0)

        roc = ROC([0, 1], [1, 0])
        assert np.isclose(roc.auc, 0.0)

    def test_auc_three_values(self):
        roc = ROC([False, True, False], [1.0, 0.0, 0.0])
        assert np.isclose(roc.auc, 0.0)

class TestROCExample3(unittest.TestCase):
    @given(gt=st.lists(st.booleans()), est=st.lists(st.floats()))
    def test_auc_hypothesis(self, gt, est):
        if np.isnan(gt).any() or np.isnan(est).any():
            with self.assertRaises(ValueError):
                ROC(gt, est)
        elif len(gt) != len(est):
            with self.assertRaises(ValueError):
                ROC(gt, est)
        elif len(gt) < 2:
            with self.assertRaises(ValueError):
                ROC(gt, est)
        else:
            roc = ROC(gt, est)
            assert 1 >= roc.auc >= 0

    @given(gt=st.lists(st.booleans()), est=st.lists(st.floats()))
    def test_roc_ran_twice(self, gt, est):
        if not np.isnan(gt).any() and not np.isnan(est).any() \
           and len(gt) == len(est) and len(gt) >= 2:
            roc = ROC(gt, est)
            fps, tps, thr = roc.roc()
            assert np.isclose(roc.tps, tps).all()
            assert np.isclose(roc.fps, fps).all()

            # Replace input with nothing, so that it's unable to compute the
            # ROC curve.
            roc.ground_truth = None
            new_fps, new_tps, new_thr = roc.roc()
            assert np.isclose(new_fps, fps).all()
            assert np.isclose(new_tps, tps).all()
            assert np.isclose(new_thr, thr).all()

    @given(gt=st.lists(st.booleans()), est=st.lists(st.floats()),
           seed=st.integers())
    def test_bootstrapping(self, gt, est, seed):
        if not np.isnan(gt).any() and not np.isnan(est).any() \
           and len(gt) == len(est) and len(gt) >= 2:
            roc = ROC(gt, est)

            if seed < 0 or seed > 2**32 - 1:
                with self.assertRaises(ValueError):
                    roc.bootstrap(seed)
            else:
                bs_roc = roc.bootstrap(seed)
                assert np.isin(bs_roc.ground_truth, roc.ground_truth).all()
                assert np.isin(bs_roc.estimates, roc.estimates).all()
