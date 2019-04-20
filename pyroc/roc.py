"""PyROC - A Python library for computing ROC curves."""

from typing import List, Optional, Tuple, Union

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

        if np.isnan(self.ground_truth).any() or np.isnan(self.estimates).any():
            raise ValueError('Ground truth or estimates contain NaN values')

        if len(self.ground_truth) != len(self.estimates):
            raise ValueError('Size of ground truth and estimates are not'
                             ' equal.')

        if len(self.ground_truth) < 2:
            raise ValueError('Ground truth and estimates cannot have size zero'
                             ' or one.')

        self.tps = None
        self.fps = None
        self.diff_values = None

    @property
    def auc(self) -> float:
        """Area Under the Curve."""
        try:
            return np.trapz(self.tps, x=self.fps)
        except IndexError:
            self.roc()
            return np.trapz(self.tps, x=self.fps)

    def roc(self) -> Tuple[np.array, np.array, np.array]:
        """Compute ROC curve."""

        if self.tps is not None and self.fps is not None \
           and self.diff_values is not None:
            return self.fps, self.tps, self.estimates[self.diff_values]

        if np.unique(self.ground_truth).shape[0] == 1:
            min_arg = np.argmin(self.estimates)
            self.fps = np.array([0., 1.])
            self.tps = np.array([1., 1.])
            self.diff_values = np.array([min_arg, min_arg])
            return self.fps, self.tps, self.estimates[self.diff_values]

        idx_sort = np.argsort(self.estimates)[::-1]
        self.ground_truth = self.ground_truth[idx_sort]
        self.estimates = self.estimates[idx_sort]

        if len(self.ground_truth) == 2:
            self.fps = np.array([0., 1.])
            self.diff_values = np.array([0, 1])
            if self.ground_truth[1] < self.ground_truth[0]:
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

    def bootstrap(self, seed: Optional[int] = None) -> 'ROC':
        """Perform bootstrap for this ROC curve."""
        rng = np.random.RandomState(seed)
        idx = np.arange(self.ground_truth.size)
        bootstrap_idx = rng.choice(idx, size=idx.shape, replace=True)
        bootstrap_ground_truth = self.ground_truth[bootstrap_idx]
        bootstrap_estimates = self.estimates[bootstrap_idx]
        return ROC(bootstrap_ground_truth, bootstrap_estimates)
