#!/usr/bin/env python
"""
Test the salary classification API.

This is not a pytest, so pytest is configured to ignore this file in
`./pytest.ini`.

Usage:
$ ./test_api.py  # Test API deployed to Heroku.
$ ./test_api.py  --l  # Test API locally.
$ ./test_api.py  --local  # Test API locally.
"""
import argparse
import json

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

# Load example.
data = CensusFeatures.Config.schema_extra["example"]
# data = {
#     "age": 40,
#     "workclass": "Self-emp-not-inc",
#     "fnlgt": 338409,
#     "education": "Masters",
#     "education_num": 14,
#     "marital_status": "Never-married",
#     "occupation": "Exec-managerial",
#     "relationship": "Not-in-family",
#     "race": "White",
#     "sex": "Female",
#     "capital_gain": 2174,
#     "capital_loss": 0,
#     "hours_per_week": 40,
#     "native_country": "United-States",
# }


def test_get():
    url = base_url
    response = requests.get(url)
    print(f"GET response:\n{response.json()}")


def test_post():
    # TODO: Add cases for both model output classes.
    url = base_url + "salary_class_inference/"
    response = requests.post(url, data=json.dumps(data))
    print(f"PUT response:\n{response.json()}")


def main():
    print()
    test_get()
    print()
    test_post()
    print()


if __name__ == "__main__":
    main()
