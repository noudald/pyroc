"""PyROC - A Python library for computing ROC curves."""

from typing import List, Union

import numpy as np


class ROC():
    """Compute the ROC curve and the AUC of the curve.

    Args:
        ground_truth: Ground truth values.
        estimates: Estimates for the ground truth values.

    """
    def __init__(self, ground_truth: Union[List[Union[int, float]], np.array],
                 estimates: Union[List[float], np.array]) -> None:
        self.ground_truth = np.array(ground_truth)
        self.estimates = np.array(estimates)

        self.tps = None
        self.fps = None
        self.diff_values = None

    @property
    def auc(self):
        """Area Under the Curve."""
        try:
            return np.trapz(self.tps, x=self.fps)
        except IndexError:
            self.roc()
            return np.trapz(self.tps, x=self.fps)

    def roc(self):
        """Compute ROC curve."""
        idx_sort = np.argsort(self.estimates)[::-1]
        self.ground_truth = self.ground_truth[idx_sort]
        self.estimates = self.estimates[idx_sort]

        tps = np.cumsum(self.ground_truth)
        fps = np.cumsum(np.max(self.ground_truth) - self.ground_truth)

        diff_values = np.append([0], np.where(np.diff(fps))[0] + 1)
        tps = tps[diff_values]
        fps = fps[diff_values]

        self.diff_values = diff_values
        self.tps = tps / np.max(tps)
        self.fps = fps / np.max(fps)

        return self.fps, self.tps, self.estimates[self.diff_values]


    def plot(self):
        """Plot ROC curve."""
        raise NotImplementedError
