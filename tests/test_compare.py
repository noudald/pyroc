import unittest

import numpy as np

from pyroc import ROC, compare_bootstrap, compare_binary

class TestCompareBootstrap(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(37)
        gt = rng.binomial(1, 0.5, 100)
        est = rng.rand((100))

        self.roc1 = ROC(gt, est)
        self.roc2 = ROC([True, True, True, False, False, False],
                        [.9, .8, .35, .4, .3, .1])

    def test_compare_bootstrap_simple(self):
        result = compare_bootstrap(self.roc1, self.roc2, seed=37)
        assert result[0]
        assert np.isclose(result[1], 0.04163033112351611)

        result = compare_bootstrap(self.roc2, self.roc1, seed=37)
        assert not result[0]
        assert np.isclose(result[1], 1 - 0.04163033112351611)

        result = compare_bootstrap(self.roc1, self.roc2, alt_hypothesis=0.01,
                                   seed=37)
        assert not result[0]

        result2 = compare_bootstrap(self.roc1, self.roc2, seed=42)
        assert result[1] != result2[1]

    def test_compare_bootstrap_extreme_alt_hypothesis(self):
        result = compare_bootstrap(self.roc1, self.roc1, alt_hypothesis=0.0,
                                   seed=37)
        assert not result[0]

        result = compare_bootstrap(self.roc2, self.roc1, alt_hypothesis=1.0,
                                   seed=37)
        assert result[1]

    def test_compare_bootstrap_failures(self):
        with self.assertRaises(ValueError):
            compare_bootstrap(self.roc1, self.roc2, seed=-1)

        for wrong_hypo in [-10, -0.01, 1.01, 10]:
            with self.assertRaises(ValueError):
                compare_bootstrap(self.roc1, self.roc2,
                                  alt_hypothesis=wrong_hypo)

class TestCompareBinary(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(37)
        gt = rng.binomial(1, 0.5, 100)
        est = rng.rand((100))

        self.roc1 = ROC(gt, est)
        self.roc2 = ROC([True, True, True, True, False, False, False],
                        [.9, .8, .7, .35, .4, .3, .1])

    def test_compare_binary_simple(self):
        result = compare_binary(self.roc1, self.roc2, seed=37)
        assert result[0]
        assert np.isclose(result[1], 0.027)

        result = compare_binary(self.roc2, self.roc1, seed=37)
        assert not result[0]
        assert np.isclose(result[1], 1 - 0.027)

        result = compare_binary(self.roc1, self.roc2, alt_hypothesis=0.01,
                                seed=37)
        assert not result[0]

        result2 = compare_binary(self.roc1, self.roc2, seed=42)
        assert result[1] != result2[1]

    def test_compare_binary_extreme_alt_hypothesis(self):
        result = compare_binary(self.roc1, self.roc1, alt_hypothesis=0.0,
                                seed=37)
        assert not result[0]

        result = compare_binary(self.roc2, self.roc1, alt_hypothesis=1.0,
                                seed=37)
        assert result[1]

    def test_compare_binary_failures(self):
        with self.assertRaises(ValueError):
            compare_binary(self.roc1, self.roc2, seed=-1)

        for wrong_hypo in [-10, -0.01, 1.01, 10]:
            with self.assertRaises(ValueError):
                compare_binary(self.roc1, self.roc2, alt_hypothesis=wrong_hypo)
