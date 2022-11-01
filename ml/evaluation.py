"""
Functions for evaluating model performance.
"""
from collections import namedtuple
from typing import Dict, Hashable, Union

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data_for_inference
from ml.model import AugmentedModel, infer, LABEL, CATEGORICAL_FEATURES


ModelMetrics = namedtuple(
    "ModelMetrics",
    "precision recall fbeta beta",
)
"""
Container for model metrics.
"""


def compute_model_metrics(
    y_reference: np.ndarray,
    y_inferred: np.ndarray,
    beta: float = 1.0,
) -> ModelMetrics:
    """
    Validates the trained machine learning model using precision, recall, and
    Fbeta.

    References:
    https://en.wikipedia.org/wiki/Precision_and_recall

    Args:
        y_reference: np.array, binarized low-noise reference labels treated as
            truth for validation or testing.
        y_inferred: np.array, binarized labels inferred by model.
        beta: fbeta parameter.

    Returns:
        A named model metrics tuple with fields
            precision: true positives / (true positives + false positives).
            recall: true positives / (true positives + false negatives).
            fbeta: weighted harmonic mean of precision and recall.
            beta: weight for `fbeta`.
    """
    precision = precision_score(y_reference, y_inferred, zero_division=1)
    recall = recall_score(y_reference, y_inferred, zero_division=1)
    fbeta = fbeta_score(y_reference, y_inferred, beta=beta, zero_division=1)
    return ModelMetrics(precision, recall, fbeta, beta)


def compute_model_metrics_for_a_single_slice(
    augmented_model: AugmentedModel,
    data_frame: pd.DataFrame,
    feature: str,
    feature_value: Union[str, int, float],
    beta: float = 1.0,
) -> ModelMetrics:
    """Compute and return model metrics for a single slice."""
    sliced_data_frame = data_frame[data_frame[feature] == feature_value]
    X, y_reference = process_data_for_inference(
        data_frame=sliced_data_frame,
        categorical_features=CATEGORICAL_FEATURES,
        input_encoder=augmented_model.input_encoder,
        label_binarizer=augmented_model.label_binarizer,
        label=LABEL,
    )
    y_inferred: np.ndarray = infer(augmented_model.model, X)
    return compute_model_metrics(y_reference, y_inferred, beta)


def compute_model_metrics_for_all_slices_of_a_feature(
    augmented_model: AugmentedModel,
    data_frame: pd.DataFrame,
    feature: str,
    beta: float = 1.0,
) -> Dict[Hashable, ModelMetrics]:
    """
    Compute and return a dictionary of model metrics for all slices of a
    particular feature.
    """
    model_metrics: Dict[Hashable, ModelMetrics] = {}
    feature_values = data_frame[feature].unique()
    for feature_value in feature_values:
        model_metrics[feature_value] = compute_model_metrics_for_a_single_slice(
            augmented_model,
            data_frame,
            feature,
            feature_value,
            beta,
        )
    return model_metrics
