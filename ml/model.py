"""
Types and functions for training and using a model.
"""
from collections import namedtuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier


AugmentedModel = namedtuple(
    "AugmentedModel",
    "model input_encoder label_binarizer",
)
"""
Container for a model together with corresponding input encoder and label
binarizer.
"""


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
