"""
Salary classification app.
"""
import os
import pickle
from typing import Optional, Union

import pandas as pd
from fastapi import FastAPI
from sklearn.ensemble import RandomForestClassifier

from census_features import CensusFeatures
from ml.data import process_data_for_inference
from ml.model import AugmentedModel, infer
from train_model import LABEL, CATEGORICAL_FEATURES, MODEL_FILE_PATH


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


augmented_model: Optional[AugmentedModel] = None
@app.on_event("startup")
async def load_model():
    """Load model on app startup, not every query."""
    with open(MODEL_FILE_PATH, 'rb') as fin:
        augmented_model = pickle.load(fin)


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
    data_frame = pd.DataFrame([features.dict()])
    data_frame.rename(
        lambda feature_name: feature_name.replace("_", "-"),
        axis="columns",
        inplace=True,
    )

    # TODO: Finish this.

    #salary_class: str = "<=50k"
    salary_class: str = ">50k"
    return {"inferred_salary_class": f"{salary_class}"}
