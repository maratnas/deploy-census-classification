"""
Types and functions for training and using a model.
"""
from collections import namedtuple
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier


LABEL = "salary"
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


AugmentedModel = namedtuple(
    "AugmentedModel",
    "model input_encoder label_binarizer",
)
"""
Container for a model together with corresponding input encoder and label
binarizer.
"""


def save_model(
    augmented_model: AugmentedModel,
    path: str = r"./models/model.pkl",
):
    """Save augmented model to disk as pickle file."""
    with open(path, 'wb') as fout:
        pickle.dump(augmented_model, fout)


def load_model(path: str = r"./models/model.pkl") -> AugmentedModel:
    """Load augmented model from pickle file."""
    with open(path, 'rb') as fin:
        augmented_model = pickle.load(fin)
    return augmented_model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """
    Fit a random forest classifier.

    Args:
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

    Args:
        model: RandomForestClassifier, trained machine learning model.
        X: np.ndarray, data for inference.

    Returns:
        y_inferred: np.ndarray, labels inferred from the model.
    """
    y_inferred = model.predict(X)
    return y_inferred
