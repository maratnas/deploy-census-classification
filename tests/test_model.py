"""
Test data processing functions and performance of saved model.
"""
import pytest
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data_for_training, process_data_for_inference
from ml.model import train_model, infer
from ml.evaluation import compute_model_metrics

from train_model import LABEL, CATEGORICAL_FEATURES


@pytest.fixture
def data_processed_for_inference(
    data_frame_test,
    augmented_model,
) -> Tuple[np.ndarray, np.ndarray]:
    X_test, y_test = process_data_for_inference(
        data_frame=data_frame_test,
        categorical_features=CATEGORICAL_FEATURES,
        input_encoder=augmented_model.input_encoder,
        label_binarizer=augmented_model.label_binarizer,
        label=LABEL,
    )
    return X_test, y_test


def test_train_model(data_frame_test):
    X_train, y_train, input_encoder, label_binarizer = \
    process_data_for_training(
        data_frame=data_frame_test,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
    )
    model: RandomForestClassifier = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_infer(augmented_model, data_processed_for_inference):
    X_test, y_test = data_processed_for_inference
    y_inferred = infer(augmented_model.model, X_test)
    assert isinstance(y_inferred, np.ndarray)


def test_compute_model_metrics(augmented_model, data_processed_for_inference):
    X_test, y_test = data_processed_for_inference
    y_inferred = infer(augmented_model.model, X_test)
    precision, recall, fbeta = compute_model_metrics(
        y_test,
        y_inferred,
    )
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_model_performance(augmented_model, data_processed_for_inference):
    X_test, y_test = data_processed_for_inference
    y_inferred = infer(augmented_model.model, X_test)
    precision, recall, fbeta = compute_model_metrics(
        y_test,
        y_inferred,
    )
    assert precision >= 0.70
    assert recall >= 0.60
    assert fbeta >= 0.65
