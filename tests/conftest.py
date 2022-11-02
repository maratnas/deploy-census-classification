"""
Package-wide pytest fixtures.
"""
import pickle

import pytest
import pandas as pd

from ml.model import AugmentedModel, load_model
from train_model import DATA_FRAME_TEST_FILE_PATH


@pytest.fixture
def data_frame_test() -> pd.DataFrame:
    with open(DATA_FRAME_TEST_FILE_PATH, 'rb') as fin:
        data_frame = pickle.load(fin)
    return data_frame


@pytest.fixture
def augmented_model() -> AugmentedModel:
    return load_model()
