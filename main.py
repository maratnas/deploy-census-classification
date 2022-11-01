"""
Salary classification app.
"""
import os
import pickle
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI
from sklearn.ensemble import RandomForestClassifier

from census_features import CensusFeatures
from ml.data import process_data_for_inference
from ml.model import AugmentedModel, infer as infer_, \
    LABEL, CATEGORICAL_FEATURES
from train_model import MODEL_FILE_PATH


# Set up DVC on Heroku.
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system(f"dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


# Instantiate app.
# Metadata fields: https://fastapi.tiangolo.com/tutorial/metadata/
app = FastAPI(
    title="Salary Classifier API",
    description="API for querying an ML model that predicts salary classes from census features.",
    version="1.0.0-alpha",
)


startup_items: Dict[str, Any] = {}
@app.on_event("startup") # => no responses until model has finished loading.
async def load_model():
    """Load model once on app startup, not every query."""
    with open(MODEL_FILE_PATH, 'rb') as fin:
        startup_items["augmented_model"] = pickle.load(fin)


@app.get("/")
async def greet():
    return {"greeting": "Welcome to the Salary Classifier API!"}


@app.post("/salary_class_inference")
async def infer(features: CensusFeatures):
    """
    Use ML model to respond to salary class inference requests.

    Args:
        features: census features.

    Returns:
        Salary classification in {"<=50k", ">50k"}.
    """
    augmented_model: AugmentedModel = startup_items["augmented_model"]
    data_frame = pd.DataFrame([features.dict()])
    data_frame.rename(
        lambda feature_name: feature_name.replace("_", "-"),
        axis="columns",
        inplace=True,
    )
    X: np.ndarray
    X, _ = process_data_for_inference(
        data_frame=data_frame,
        categorical_features=CATEGORICAL_FEATURES,
        input_encoder=augmented_model.input_encoder,
        label_binarizer=augmented_model.label_binarizer,
    )

    # `inferred_class`` takes a value in {"<=50k", ">50k"}.
    y_inferred = infer_(augmented_model.model, X)
    inferred_class: str = augmented_model.label_binarizer.inverse_transform(
        y_inferred,
    )[0]

    #inferred_class: str = "<=50k"  # Place-holder for testing only.
    return {"inferred_salary_class": f"{inferred_class}"}
