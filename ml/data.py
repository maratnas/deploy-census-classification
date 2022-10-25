"""
Data manipulation utilities.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X: pd.DataFrame,
    categorical_features: List[str] = [],
    label: Optional[str] = None,
    training: bool = True,
    encoder: Optional[OneHotEncoder] = None,
    label_binarizer: Optional[LabelBinarizer]=None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[OneHotEncoder],
    Optional[LabelBinarizer],
]:
    """
    Process the data used in the machine learning pipeline.

    Processes the data using one-hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in
    functionality that scales the continuous data.

    Args:
        X: pd.DataFrame, Dataframe containing the features and label. Columns in
            `categorical_features`.

        categorical_features: list[str], List containing the names of the
            categorical features (default=[]).

        label: str, Name of the label column in `X`. If None, then an empty
            array will be returned for y (default=None).

        training: bool, Indicator if training mode or inference/validation mode.

        encoder: sklearn.preprocessing._encoders.OneHotEncoder, Trained sklearn
            OneHotEncoder, only used if training=False.

        label_binarizer: sklearn.preprocessing._label.LabelBinarizer, Trained
            sklearn LabelBinarizer, only used if training=False.

    Returns:
        X : np.ndarray, Processed data.

        y : np.ndarray, Processed labels if labeled=True, otherwise empty
            np.ndarray.

        encoder : sklearn.preprocessing._encoders.OneHotEncoder, trained
            OneHotEncoder if training is True, otherwise returns the encoder
            passed in.

        label_binarizer : sklearn.preprocessing._label.LabelBinarizer, trained
            LabelBinarizer if training is True, otherwise returns the binarizer
            passed in.
    """
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        label_binarizer = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = label_binarizer.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = label_binarizer.transform(y.values).ravel()
        # Handle y is None case because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, label_binarizer
