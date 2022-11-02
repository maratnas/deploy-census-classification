#!/usr/bin/env python
"""
Manual test of the salary classification API.

For this script to work with `--local`, make sure to start the server.
$ uvicorn main:app --reload

Usage:
$ ./query_live_api.py  # Test API deployed to Heroku.
$ ./query_live_api.py  -l  # Test API locally.
$ ./query_live_api.py  --local  # Test API locally.
"""
import argparse
import json
from typing import Dict, Any

import requests

from census_features import CensusFeatures


# Parse command arguments.
parser = argparse.ArgumentParser(
    prog="Test salary classification API.",
)
parser.add_argument('-l', '--local', action='store_true')
args = parser.parse_args()
if args.local:
    base_url = "http://127.0.0.1:8000/"
else:
    base_url = "https://census-classification-a4a401cd.herokuapp.com/"


# Example features.
#data = CensusFeatures.Config.schema_extra["example"]
example_low_salary = {
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
example_high_salary = {
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


def test_get():
    """Confirm that a greeting is returned."""
    url = base_url
    response = requests.get(url)
    print(
        "GET\n"
        f"response: {response.json()}\n"
        f"status_code: {response.status_code}\n"
    )


def test_post_low_salary(example_low_salary: Dict[str, Any]):
    """Confirm low salary class is correctly returned."""
    url = base_url + "salary_class_inference/"
    response = requests.post(url, data=json.dumps(example_low_salary))
    response_json = response.json()
    assert response.status_code == 200
    assert response_json["inferred_salary_class"] == "<=50K"
    print(
        "POST\n"
        f"response: {response.json()}\n"
        f"status_code: {response.status_code}\n"
    )


def test_post_high_salary(example_high_salary: Dict[str, Any]):
    """Confirm high salary class is correctly returned."""
    url = base_url + "salary_class_inference/"
    response = requests.post(url, data=json.dumps(example_high_salary))
    response_json = response.json()
    assert response.status_code == 200
    assert response_json["inferred_salary_class"] == ">50K"
    print(
        "POST\n"
        f"response: {response.json()}\n"
        f"status_code: {response.status_code}\n"
    )


def main():
    """Run tests and print results."""
    print()
    test_get()

    print()
    test_post_low_salary(example_low_salary)

    print()
    test_post_high_salary(example_high_salary)

    print()


if __name__ == "__main__":
    main()
