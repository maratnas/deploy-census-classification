#!/usr/bin/env python
"""
Load data, train a machine learning model, and save the model to disk.
"""

import pickle

from re import M
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import CATEGORICAL_FEATURES, process_data
from ml.model import train_model, infer, compute_model_metrics


LABEL = "salary"
DATA_FILE_PATH = r"./data/census-cooked.csv"
MODEL_FILE_PATH = r"./models/model.pkl"


def main() -> None:
    # Load data from file.
    data_frame: pd.DataFrame = pd.read_csv(DATA_FILE_PATH)

    data_frame_train, data_frame_test = train_test_split(
        data_frame,
        test_size=0.20,
    )

    X_train, y_train, encoder, lb = process_data(
        X=data_frame_train,
        categorical_features=CATEGORICAL_FEATURES,
        label="salary",
        training=True,
    )

    X_test, y_test, *_ = process_data(
        X=data_frame_test,
        categorical_features=CATEGORICAL_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train model.
    model: RandomForestClassifier = train_model(X_train, y_train)

    # Save model.
    with open(MODEL_FILE_PATH, 'wb') as fout:
        pickle.dump(model, fout)

    # Evaluate on test set.
    with open(MODEL_FILE_PATH, 'rb') as fin:
        model = pickle.load(fin)
    inferred_labels = infer(model, X_test)
    precision, recall, fbeta = compute_model_metrics(
        y_test, inferred_labels)
    print("\nprecision:", precision)
    print("recall:", recall)
    print("fbeta:", fbeta)
    print()


if __name__ == "__main__":
    main()