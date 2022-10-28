"""
Data manipulation utilities.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data_for_training(
    data_frame: pd.DataFrame,
    categorical_features: List[str],
    label: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    OneHotEncoder,
    LabelBinarizer,
]:
    """
    Process data for training in the machine learning pipeline.

    One-hot encoding is used for the categorical input features and a binarizer
    for the labels.

    Args:
        data_frame: features and labels. Columns should include
            `categorical_features` and `label`.
        categorical_features: names of the categorical features.
        label: label column name.

    Returns:
        X: np.ndarray, Processed data.
        y: np.ndarray, Processed labels.
        input_encoder: trained OneHotEncoder.
        label_binarizer: trained LabelBinarizer.
    """
    y: pd.Series = data_frame[label]
    data_frame: pd.DataFrame = data_frame.drop([label], axis=1)

    # np.ndarray of strings (dtype = "O" for "Object").
    X_categorical: np.ndarray = data_frame[categorical_features].values

    # pd.DataFrame of numeric values (dtypes = "int64").
    X_continuous: np.DataFrame = \
        data_frame.drop(*[categorical_features], axis=1)

    input_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    # np.ndarray of numeric values (dtype = "float64") in [0.0, 1.0].
    X_categorical = input_encoder.fit_transform(X_categorical)

    label_binarizer = LabelBinarizer()
    # np.ndarray of numeric values (dtype = "int64") in [0.0, 1.0].
    y = label_binarizer.fit_transform(y.values).ravel()

    X: np.ndarray = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, input_encoder, label_binarizer


def process_data_for_inference(
    data_frame: pd.DataFrame,
    categorical_features: List[str],
    input_encoder: OneHotEncoder,
    label_binarizer: LabelBinarizer,
    label: Optional[str] = None,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
]:
    """
    Process data for inference in the machine learning pipeline, possibly with
    validation.

    One-hot encoding is used for the categorical input features and a binarizer
    for the labels.

    Args:
        data_frame: features and label. Columns should include
            `categorical_features`, and `label` if not None.
        categorical_features: names of the categorical features.
        input_encoder: trained OneHotEncoder.
        label_binarizer: trained LabelBinarizer.
        label: label column name. None => return None for `y`.

    Returns:
        X : Processed data.
        y : Processed labels if `label` provided, else None.
    """
    if label:
        y: pd.Series = data_frame[label]
        data_frame: pd.DataFrame = data_frame.drop([label], axis=1)
    else:
        y = None

    # np.ndarray of strings (dtype = "O" for "Object").
    X_categorical: np.ndarray = data_frame[categorical_features].values

    # pd.DataFrame of numeric values (dtypes = "int64").
    X_continuous: np.DataFrame = \
        data_frame.drop(*[categorical_features], axis=1)

    # np.ndarray of numeric values (dtype = "float64") in [0.0, 1.0].
    X_categorical = input_encoder.transform(X_categorical)

    if y is not None:
        # This will fail if the dataset is unlabeled.
        try:
            # np.ndarray of numeric values (dtype = "int64") in {0, 1}.
            y = label_binarizer.transform(y.values).ravel()
        except AttributeError:
            pass

    X: np.ndarray = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y
