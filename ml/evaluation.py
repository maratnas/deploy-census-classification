"""
Functions for evaluating model performance.
"""
from typing import Tuple

import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score


def compute_model_metrics(
    y_reference: np.ndarray,
    y_inferred: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Args:
        y_reference: np.array, binarized low-noise reference labels treated as
            truth for validation or testing.
        y_inferred: np.array, binarized labels inferred by model.

    Returns:
        precision: float
        recall: float
        fbeta: float
    """
    fbeta = fbeta_score(y_reference, y_inferred, beta=1, zero_division=1)
    precision = precision_score(y_reference, y_inferred, zero_division=1)
    recall = recall_score(y_reference, y_inferred, zero_division=1)
    return precision, recall, fbeta
