#!/usr/bin/env python
"""
Test the salary classification API.

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
    url = "http://127.0.0.1:8000/salary_class_inference/"
else:
    url = "https://census-classification-a4a401cd.herokuapp.com/salary_class_inference/"

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

response = requests.post(url, data=json.dumps(data))

print(response.json())
