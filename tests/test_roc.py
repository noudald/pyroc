import numpy as np

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
