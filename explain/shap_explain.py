import numpy as np
from typing import Callable, Optional, List


def shap_explain_features(
    predict_fn: Callable,
    X_background: np.ndarray,
    X_instance: np.ndarray,
    n_samples: int = 100,
) -> np.ndarray:
    shap_values = np.zeros(X_instance.shape[1])

    for _ in range(n_samples):
        mask = np.random.binomial(1, 0.5, size=X_instance.shape[1])
        X_masked = X_instance.copy()
        for j in range(X_instance.shape[1]):
            if mask[j] == 0:
                idx = np.random.randint(len(X_background))
                X_masked[0, j] = X_background[idx, j]

        baseline_pred = predict_fn(X_masked)[0]
        for j in range(X_instance.shape[1]):
            X_plus = X_masked.copy()
            X_minus = X_masked.copy()
            X_plus[0, j] = X_instance[0, j]
            X_minus[0, j] = X_background[np.random.randint(len(X_background)), j]
            shap_values[j] += (predict_fn(X_plus)[0] - predict_fn(X_minus)[0]) * (2 * mask[j] - 1)

    return shap_values / n_samples
