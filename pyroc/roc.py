"""PyROC - A Python library for computing ROC curves."""

from collections import namedtuple
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# from pyroc import bootstrap_roc

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

    def plot(self,
             x_label: str = '1 - Specificity',
             y_label: str = 'Sensitivity',
             title: str = 'ROC Curve',
             color: str = 'blue',
             bootstrap: bool = False,
             num_bootstraps: int = 1000,
             num_bootstrap_jobs: int = 1,
             seed: Optional[int] = None,
             p_value: float = 0.05,
             mean_roc: bool = False,
             show_min_max: bool = False,
             ax: plt.Axes = None) -> plt.Axes:
        """Plot ROC curve."""

        # TODO: Clean up method, reformat code into smaller methods.
        # pylint: disable=too-many-locals

        if not ax:
            ax = plt.gca()

        if bootstrap:
            # Import bootstrap_roc locally to avoid cross reference imports.
            from pyroc import bootstrap_roc
            bs_roc_list = bootstrap_roc(self, num_bootstraps=num_bootstraps,
                                        seed=seed, n_jobs=num_bootstrap_jobs)

            arange = np.arange(0, 1.01, 0.01)
            interp_list = []
            for cur_roc in bs_roc_list:
                cur_fps, cur_tps, _ = cur_roc.roc()
                interp_list.append(np.interp(arange, cur_fps, cur_tps))
            interp_funcs = np.vstack(interp_list)

            BootstrapPlot = namedtuple(
                'BootstrapPlot',
                ['xrange', 'min', 'max', 'mean', 'min_quantile', 'max_quantile'])

            bsp = BootstrapPlot(
                xrange=arange,
                min=np.min(interp_funcs, axis=0) if show_min_max else None,
                max=np.max(interp_funcs, axis=0) if show_min_max else None,
                mean=np.mean(interp_funcs, axis=0) if mean_roc else None,
                min_quantile=np.quantile(interp_funcs, p_value / 2, axis=0),
                max_quantile=np.quantile(interp_funcs, 1 - p_value / 2, axis=0)
            )

            ax.fill_between(bsp.xrange, bsp.min_quantile, bsp.max_quantile,
                            alpha=0.2, color=color)
            if show_min_max:
                ax.fill_between(bsp.xrange, bsp.min, bsp.max, alpha=0.1, color=color)
            ax.plot(bsp.xrange, bsp.min_quantile, color=color, alpha=0.3)
            ax.plot(bsp.xrange, bsp.max_quantile, color=color, alpha=0.3)

        if mean_roc:
            if not bootstrap:
                raise RuntimeError(
                    'Cannot plot mean ROC curve without bootstrapping.')
            ax.plot(bsp.xrange, bsp.mean, color=color)
        else:
            fps, tps, _ = self.roc()
            ax.plot(fps, tps, color=color)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return ax
