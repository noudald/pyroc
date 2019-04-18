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
