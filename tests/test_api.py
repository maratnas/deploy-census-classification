"""
Test the salary classification API.
"""
import json
import pytest
from typing import Dict, Union

from fastapi.testclient import TestClient

from census_features import CensusFeatures
from main import app


# Globally defining a client with `client = TestClient(app)` does not correctly
# trigger startup events, so instead we use this fixture to get the client.
# https://github.com/tiangolo/fastapi/issues/1072#issuecomment-612942829
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def example_low_salary() -> Dict[str, Union[str, int, float]]:
    return {
        "age": 16,
        "workclass": "Never-worked",
        "fnlgt": 77516,
        "education": "Preschool",
        "education_num": 1,
        "marital_status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 1411,
        "hours_per_week": 5,
        "native_country": "United-States",
    }


@pytest.fixture
def example_high_salary() -> Dict[str, Union[str, int, float]]:
    return {
        "age": 40,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 338409,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 5060,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }


def test_get(client):
    """Confirm that a greeting is returned."""
    url = "/"
    response = client.get(url)
    response_json = response.json()
    assert response.status_code == 200
    assert response_json["greeting"] == "Welcome to the Salary Classifier API!"


def test_post_low_salary(client, example_low_salary):
    """Confirm low salary class is correctly returned."""
    url = "/salary_class_inference"
    response = client.post(url, json=example_low_salary)
    response_json = response.json()
    assert response.status_code == 200
    assert response_json["inferred_salary_class"] == "<=50K"


def test_post_high_salary(client, example_high_salary):
    """Confirm high salary class is correctly returned."""
    url = "/salary_class_inference"
    response = client.post(url, json=example_high_salary)
    response_json = response.json()
    assert response.status_code == 200
    assert response_json["inferred_salary_class"] == ">50K"
