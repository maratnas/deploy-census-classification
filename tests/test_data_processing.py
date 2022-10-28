"""
Test data processing functions.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from ml.data import process_data_for_training, process_data_for_inference
from train_model import LABEL, CATEGORICAL_FEATURES


def test_process_data_for_training(data_frame_test):
    X_train, y_train, input_encoder, label_binarizer = \
    process_data_for_training(
        data_frame=data_frame_test,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
    )
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(input_encoder, OneHotEncoder)
    assert isinstance(label_binarizer, LabelBinarizer)


def test_process_data_for_inference(data_frame_test, augmented_model):
    X_test, y_test = process_data_for_inference(
        data_frame=data_frame_test,
        categorical_features=CATEGORICAL_FEATURES,
        input_encoder=augmented_model.input_encoder,
        label_binarizer=augmented_model.label_binarizer,
        label=LABEL,
    )
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
