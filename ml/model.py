import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train: np.ndarray, y_train: np.ndarray):
    """
    Trains a machine learning model and returns it.

    Inputs:
        X_train : np.array, Training data.
        y_train : np.array, Labels.

    Returns:
        model: Trained machine learning model.
    """

    pass


def compute_model_metrics(y: np.ndarray, preds: np.ndarray):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs:
        y : np.array, Known labels, binarized.
        preds : np.array, Predicted labels, binarized.

    Returns:
        precision : float
        recall : float
        fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def infer(model, X: np.ndarray):
    """
    Run model inferences and return the predictions.

    Inputs:
        model : ???, Trained machine learning model.
        X : np.ndarray, Data used for prediction.

    Returns:
        preds : np.ndarray, predictions from the model.
    """
    pass
