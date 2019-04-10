"""PyROC - A Python library for computing ROC curves."""


class ROC():
    """Compute the ROC curve and the AUC of the curve.

    Args:
        ground_truth: Ground truth values.
        estimates: Estimates for the ground truth values.

    """
    def __init__(self, ground_truth, estimates):
        self.ground_truth = ground_truth
        self.estimates = estimates

    @property
    def auc(self):
        """Area Under the Curve."""
        raise NotImplementedError

    def plot(self):
        """Plot ROC curve."""
        raise NotImplementedError
