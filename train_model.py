#!/usr/bin/env python
"""
Train machine learning model.
"""

from sklearn.model_selection import train_test_split

# Add necessary imports for starter code.

# Add code to load in data.

# Optional enhancement, use K-fold cross validation instead of a train-test
# split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Proces test data with `process_data`` function.

# Train and save a model.
