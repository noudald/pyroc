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
        self.ground_truth = np.array(ground_truth).astype(np.int)
        self.estimates = np.array(estimates).astype(np.float)

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

        if len(self.ground_truth) != len(self.estimates):
            raise ValueError('Size of ground truth and estimates are not'
                             ' equal.')

        if len(self.ground_truth) < 2:
            raise ValueError('Ground truth and estimates cannot have size zero'
                             ' or one.')

        if np.unique(self.ground_truth).shape[0] == 1:
            min_th = np.min(self.estimates)
            self.fps = np.array([0., 1.])
            self.tps = np.array([1., 1.])
            return self.fps, self.tps, np.array([min_th, min_th])

        idx_sort = np.argsort(self.estimates)[::-1]
        self.ground_truth = self.ground_truth[idx_sort]
        self.estimates = self.estimates[idx_sort]

        if len(self.ground_truth) == 2:
            self.fps = np.array([0., 1.])
            if self.estimates[1] < self.estimates[0]:
                self.tps = np.array([1., 1.])
            else:
                self.tps = np.array([0., 0.])
            return self.fps, self.tps, self.estimates

        tps = np.cumsum(self.ground_truth)
        fps = np.cumsum(np.max(self.ground_truth) - self.ground_truth)


        diff_values = np.append([0], np.where(np.diff(fps))[0] + 1)
        self.tps = tps[diff_values]
        self.fps = fps[diff_values]

        self.diff_values = diff_values
        if not np.isclose(np.max(self.tps), 0.0):
            self.tps = self.tps / np.max(self.tps)
        if not np.isclose(np.max(self.fps), 0.0):
            self.fps = self.fps / np.max(self.fps)

        return self.fps, self.tps, self.estimates[self.diff_values]


    def plot(self):
        """Plot ROC curve."""
        raise NotImplementedError
