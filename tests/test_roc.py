import unittest

from itertools import product

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

class TestROCBootstrap(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(37)
        num = 100
        gt = rng.binomial(1, 0.5, num)
        est = rng.rand((num))
        self.roc = ROC(gt, est)

    def test_bootstrap_confidence_default(self):
        bsp = self.roc.bootstrap_confidence()

        assert bsp.xrange.size == 101
        assert not bsp.min
        assert not bsp.max
        assert not bsp.mean
        assert bsp.min_quantile.size == 101
        assert bsp.max_quantile.size == 101
        assert (bsp.min_quantile <= bsp.max_quantile).all()

    def test_bootstrap_confidence_num_bootstraps(self):
        for num_bootstraps in [-1000, -10, -1, 0, 1, 10, 100, 1000]:
            if num_bootstraps <= 0:
                with self.assertRaises(ValueError):
                    self.roc.bootstrap_confidence(num_bootstraps=num_bootstraps)
            else:
                bsp = self.roc.bootstrap_confidence(
                    num_bootstraps=num_bootstraps)

                assert bsp.xrange.size == 101
                assert not bsp.min
                assert not bsp.max
                assert not bsp.mean
                assert bsp.min_quantile.size == 101
                assert bsp.max_quantile.size == 101
                assert (bsp.min_quantile <= bsp.max_quantile).all()

    def test_bootstrap_confidence_num_bootstrap_jobs(self):
        for num_jobs_1, num_jobs_2 in product([-1, 1, 4], repeat=2):
            bsp1 = self.roc.bootstrap_confidence(
                num_bootstrap_jobs=num_jobs_1, seed=37)
            bsp2 = self.roc.bootstrap_confidence(
                num_bootstrap_jobs=num_jobs_2, seed=37)

            assert np.isclose(bsp1.min_quantile, bsp2.min_quantile).all()
            assert np.isclose(bsp1.max_quantile, bsp2.max_quantile).all()

    def test_bootstrap_confidence_show_min_max(self):
        bsp1 = self.roc.bootstrap_confidence(show_min_max=True, seed=37)

        assert bsp1.xrange.size == 101
        assert bsp1.min.size == 101
        assert bsp1.max.size == 101
        assert not bsp1.mean
        assert bsp1.min_quantile.size == 101
        assert bsp1.max_quantile.size == 101
        assert (bsp1.min_quantile <= bsp1.max_quantile).all()

        bsp2 = self.roc.bootstrap_confidence(show_min_max=False, seed=37)

        assert not bsp2.min
        assert not bsp2.max
        assert np.isclose(bsp1.min_quantile, bsp2.min_quantile).all()
        assert np.isclose(bsp1.max_quantile, bsp2.max_quantile).all()

    def test_bootstrap_confidence_mean_roc(self):
        bsp1 = self.roc.bootstrap_confidence(mean_roc=True, seed=37)

        assert bsp1.xrange.size == 101
        assert not bsp1.min
        assert not bsp1.max
        assert bsp1.mean.size == 101
        assert bsp1.min_quantile.size == 101
        assert bsp1.max_quantile.size == 101
        assert (bsp1.min_quantile <= bsp1.max_quantile).all()

        bsp2 = self.roc.bootstrap_confidence(mean_roc=False, seed=37)

        assert not bsp2.mean
        assert np.isclose(bsp1.min_quantile, bsp2.min_quantile).all()
        assert np.isclose(bsp1.max_quantile, bsp2.max_quantile).all()

        bsp3 = self.roc.bootstrap_confidence(
            mean_roc=True, show_min_max=True, seed=37)

        assert (bsp3.min <= bsp3.mean).all()
        assert (bsp3.mean <= bsp3.max).all()

    def test_bootstrap_confidence_p_value(self):
        for p_value in [-1, -0.001, 1, 1.001, 10]:
            with self.assertRaises(ValueError):
                self.roc.bootstrap_confidence(p_value=p_value)

        bsp1 = self.roc.bootstrap_confidence(p_value=.5, seed=37)
        bsp2 = self.roc.bootstrap_confidence(p_value=.1, seed=37)
        bsp3 = self.roc.bootstrap_confidence(p_value=0, seed=37)

        assert (bsp3.min_quantile <= bsp2.min_quantile).all()
        assert (bsp2.min_quantile <= bsp1.min_quantile).all()
        assert (bsp1.min_quantile <= bsp1.max_quantile).all()
        assert (bsp1.max_quantile <= bsp2.max_quantile).all()
        assert (bsp2.max_quantile <= bsp3.max_quantile).all()

    def test_plot_with_bootstrap(self):
        """Could be tested better, any improvements are welcome."""
        ax = self.roc.plot(bootstrap=True)
        assert ax

        ax = self.roc.plot(bootstrap=True, show_min_max=True)
        assert ax

        ax = self.roc.plot(bootstrap=True, show_min_max=True, label='test')
        assert ax

        ax = self.roc.plot(bootstrap=True, mean_roc=True)
        assert ax

        ax = self.roc.plot(bootstrap=True, mean_roc=True, label='test')
        assert ax

        with self.assertRaises(RuntimeError):
            self.roc.plot(bootstrap=False, mean_roc=True)

        ax = self.roc.plot(bootstrap=False)
        assert ax


class TestROCCompare(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(37)
        gt = rng.binomial(1, 0.5, 100)
        est = rng.rand((100))

        self.roc1 = ROC(gt, est)
        self.roc2 = ROC([True, True, True, False, False, False],
                        [.9, .8, .35, .4, .3, .1])
        self.roc3 = ROC([True, True, True, False, False, False],
                        [.9, .8, .35, .4, .3, .1],
                        statistical_power=0.99)

    def test_equal(self):
        assert self.roc1 == self.roc1
        assert self.roc2 == self.roc2
        assert self.roc2 == self.roc3
        assert not self.roc1 == self.roc2
        assert not self.roc1 == self.roc3

        with self.assertRaises(NotImplementedError):
            self.roc1 == 10

    def test_unequal(self):
        assert self.roc1 < self.roc2
        assert not (self.roc1 > self.roc2)
        assert self.roc2 > self.roc1
        assert not (self.roc2 < self.roc1)
        assert self.roc1 <= self.roc2
        assert not (self.roc1 >= self.roc2)
        assert self.roc2 >= self.roc1
        assert not (self.roc2 <= self.roc1)

        # ROC1 is not smaller than ROC3, because of the statistical power.
        assert not (self.roc1 < self.roc3)
        assert not (self.roc3 > self.roc3)
        assert not (self.roc1 <= self.roc3)
        assert not (self.roc3 >= self.roc3)

        with self.assertRaises(NotImplementedError):
            self.roc1 > 10

        with self.assertRaises(NotImplementedError):
            self.roc1 >= 10

        with self.assertRaises(NotImplementedError):
            self.roc1 < 10

        with self.assertRaises(NotImplementedError):
            self.roc1 <= 10
