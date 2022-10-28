#!/usr/bin/env python
"""
Load data, train a machine learning model, and save the model to disk.
"""
import pickle

# from re import M
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from ml.data import process_data_for_training, process_data_for_inference
from ml.model import AugmentedModel, train_model, infer, compute_model_metrics


DATA_FILE_PATH = r"./data/census-clean.csv"
MODEL_FILE_PATH = r"./models/model.pkl"
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


def main() -> None:
    data_frame: pd.DataFrame = pd.read_csv(DATA_FILE_PATH)

    data_frame_train, data_frame_test = train_test_split(
        data_frame,
        test_size=0.20,
    )

    X_train, y_train, input_encoder, label_binarizer = \
    process_data_for_training(
        data_frame=data_frame_train,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
    )

    X_test, y_test = process_data_for_inference(
        data_frame=data_frame_test,
        categorical_features=CATEGORICAL_FEATURES,
        input_encoder=input_encoder,
        label_binarizer=label_binarizer,
        label=LABEL,
    )

    # Train model.
    model: RandomForestClassifier = train_model(X_train, y_train)
    augmented_model = AugmentedModel(model, input_encoder, label_binarizer)

    # Save augmented model.
    with open(MODEL_FILE_PATH, 'wb') as fout:
        pickle.dump(augmented_model, fout)

    # Load saved augmented model and evaluate on test set.
    with open(MODEL_FILE_PATH, 'rb') as fin:
        augmented_model = pickle.load(fin)
    y_inferred = infer(augmented_model.model, X_test)
    precision, recall, fbeta = compute_model_metrics(
        y_test,
        y_inferred,
    )
    print(f"\nprecision: {precision}")
    print(f"recall: {recall}")
    print(f"fbeta: {fbeta}\n")


if __name__ == "__main__":
    main()
