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
        X_train : np.array, training data.
        y_train : np.array, labels.

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
        X : np.ndarray, data used for inference.

    Returns:
        inferred_labels: np.ndarray, labels inferred from the model.
    """
    inferred_labels = model.predict(X)
    return inferred_labels


def compute_model_metrics(
    y: np.ndarray,
    preds: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs:
        y : np.array, known labels, binarized.
        preds : np.array, predicted labels, binarized.

    Returns:
        precision : float
        recall : float
        fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta
