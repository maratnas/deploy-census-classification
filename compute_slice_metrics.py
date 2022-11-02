#!/usr/bin/env python
"""
Load data, compute model metrics for all slices of a particular feature, and
save results to a file.
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
from ml.evaluation import compute_model_metrics_for_all_slices_of_a_feature, \
    ModelMetrics
from train_model import DATA_FRAME_TEST_FILE_PATH, DATA_FILE_PATH


SLICE_METRICS_FILE_PATH = "slice_output.txt"
FEATURE = "sex"


def main() -> None:
    # Load saved augmented model.
    augmented_model = load_model()

    # Load test data frame.
    with open(DATA_FRAME_TEST_FILE_PATH, 'rb') as fin:
        data_frame_test = pickle.load(fin)

    # Compute metrics over all slices of FEATURE.
    metrics: Dict[Hashable, ModelMetrics] = \
            compute_model_metrics_for_all_slices_of_a_feature(
        augmented_model,
        data_frame_test,
        feature=FEATURE,
        beta=1.0,
    )

    # Write slice metrics to file.
    with open(SLICE_METRICS_FILE_PATH, 'wt') as fout:
        fout.write(f"_Metrics for all slices of feature \"{FEATURE}\"_\n")
        for key, value in metrics.items():
            fout.write(f"\n{FEATURE} = {key}")
            fout.write(f"\n  precision: {value.precision}")
            fout.write(f"\n  recall: {value.recall}")
            fout.write(f"\n  F{value.beta}: {value.fbeta}\n")
        fout.write("\n")


if __name__ == "__main__":
    main()
