"""Utility functions for metric evaluation in forecasting strategies."""

from typing import Any, List

import numpy as np
import pandas as pd

from tsfb.base.evaluation.metrics.metrics import get_instantiated_metric_dict


def evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaler: Any,
    metric_names: List[str],
) -> List[float]:
    """Compute evaluation metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        scaler: Scaler used for inverse normalization
        metric_names: List of metric names to compute

    Returns:
        List of computed metric scores
    """
    df = pd.DataFrame({"label": y_true.flatten(), "predicted": y_pred.flatten()})

    metric_dict = get_instantiated_metric_dict()
    results: List[float] = []

    for name in metric_names:
        metric = metric_dict.get(name)
        if metric is None:
            raise ValueError(f"Metric '{name}' is not recognized.")
        # Always pass scaler, metric itself will decide whether to use it
        res = metric.compute_scores(df, "label", "predicted", scaler=scaler)
        results.append(res)

    return results
