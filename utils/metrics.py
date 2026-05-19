import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score, median_absolute_error, max_error,
)
from typing import Dict, List, Optional


def comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_std: Optional[np.ndarray] = None) -> Dict:
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    pct_errors = np.abs(errors) / (np.abs(y_true) + 1e-8) * 100

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
        "mape": float(np.mean(pct_errors)),
        "mae_std": float(abs_errors.std()),
        "mae_p25": float(np.percentile(abs_errors, 25)),
        "mae_p50": float(np.percentile(abs_errors, 50)),
        "mae_p75": float(np.percentile(abs_errors, 75)),
        "mae_p90": float(np.percentile(abs_errors, 90)),
        "bias": float(errors.mean()),
        "error_std": float(errors.std()),
    }

    if y_std is not None:
        z_scores = abs_errors / (y_std + 1e-8)
        metrics["within_1std"] = float((z_scores <= 1).mean())
        metrics["within_2std"] = float((z_scores <= 2).mean())
        nll = 0.5 * np.log(2 * np.pi * y_std ** 2) + (errors ** 2) / (2 * y_std ** 2 + 1e-8)
        metrics["negative_log_likelihood"] = float(nll.mean())

    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]
    for t in thresholds:
        metrics[f"accuracy_within_{t}d"] = float((abs_errors <= t).mean())

    metrics["worst_5pct_mae"] = float(np.mean(np.sort(abs_errors)[-max(1, len(abs_errors)//20):]))

    return metrics
