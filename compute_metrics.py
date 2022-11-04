#!/usr/bin/env python
"""
Load data, compute and print model metrics over entire test set.
"""
import pickle
from typing import Dict, Hashable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from ml.data import process_data_for_inference
from ml.model import AugmentedModel, infer, load_model, \
    LABEL, CATEGORICAL_FEATURES
from ml.evaluation import compute_model_metrics, ModelMetrics
from train_model import DATA_FRAME_TEST_FILE_PATH, DATA_FILE_PATH


def main() -> None:
    # Load saved augmented model.
    augmented_model = load_model()

    # Load test data frame.
    with open(DATA_FRAME_TEST_FILE_PATH, 'rb') as fin:
        data_frame_test = pickle.load(fin)

    # Preprocess test data.
    X_test, y_test = process_data_for_inference(
        data_frame=data_frame_test,
        categorical_features=CATEGORICAL_FEATURES,
        input_encoder=augmented_model.input_encoder,
        label_binarizer=augmented_model.label_binarizer,
        label=LABEL,
    )

    # Infer.
    y_inferred = infer(augmented_model.model, X_test)

    # Compute metrics over entire test set.
    precision, recall, fbeta, beta = compute_model_metrics(
        y_test,
        y_inferred,
        beta=1.0,
    )

    print(f"\nprecision: {precision:.4}")
    print(f"recall: {recall:.4}")
    print(f"F{beta}: {fbeta:.4}\n")


if __name__ == "__main__":
    main()
