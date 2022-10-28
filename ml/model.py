from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """
    Fit a random forest classifier.

    Inputs:
        X_train: np.array, training data.
        y_train: np.array, labels.

    Returns:
        model: trained random forest classifier.
    """
    classifier = RandomForestClassifier(n_estimators=20)
    classifier.fit(X_train, y_train)
    return classifier


def infer(
    model: RandomForestClassifier,
    X: np.ndarray,
) -> np.ndarray:
    """
    Run model inferences and return inferred labels.

    Inputs:
        model: RandomForestClassifier, trained machine learning model.
        X: np.ndarray, data for inference.

    Returns:
        y_inferred: np.ndarray, labels inferred from the model.
    """
    y_inferred = model.predict(X)
    return y_inferred


def compute_model_metrics(
    y_reference: np.ndarray,
    y_inferred: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs:
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
