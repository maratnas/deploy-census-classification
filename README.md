# Salary Classification Pipeline

This project is an implementation of a scalable ML pipeline for infering a
person's salary class ("<=50K" or ">50K") based on demographic features found in
census data. The ML model is a random forest trained in Python with
scikit-learn. Code is tracked with Git. Data and model artifacts are tracked
with DVC. An Amazon S3 bucket serves as a remote DVC store. CI and CD are
implemented with GitHub Actions in concert with the Heroku PaaS. Once deployed
to Heroku, the app serves a salary classification API.


## URLs

GitHub repository  
https://github.com/maratnas/deploy-census-classification  
  
Heroku Dashboard  
https://dashboard.heroku.com/apps/census-classification-a4a401cd/  
  
Heroku API Root  
https://census-classification-a4a401cd.herokuapp.com/docs  
  
Heroku API Docs  
https://census-classification-a4a401cd.herokuapp.com/docs  
  
Data Origin  
https://archive.ics.uci.edu/ml/datasets/census+income  
https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/data/census.csv  
  
Udacity Course Project Starter  
https://github.com/udacity/nd0821-c3-starter-code/


## Salient Files

`model_card.md`: ML model card.

`requirements.txt`: Python dependencies.

`.github/workflows/python-app.yml`: GitHub Actions specs.

`data/census.csv`: DVC-tracked raw census data.

`data/census-clean.csv`: DVC-tracked cleaned census data.

`data/data_frame_test.pkl`: DVC-tracked test data frame used by CI.

`models/model.pkl`: DVC-tracked ML model artifact.
.
`eda.ipynb`: EDA notebook.

`train_model.py`: Script for training and serializing the ML model.

`main.py`: App that serves the API on Heroku.

`census_features.py`: Defines `CensusFeatures` type for posting to the API.

`compute_metrics.py`: Script for evaluating ML model performance on the full
dataset.

`compute_slice_metrics.py`: Script for evaluationg ML model performance on
dataset slices.

`ml/data.py`: Provides data processing functions.

`ml/evaluation.py`: Provides functions for evaluating ML model performance.

`ml/model.py`: Provides training and inference functions, and the
  `AugmentedModel` type for serializing models.

`tests/data_processing.py`: PyTest test of data processing functions. Gets run
with CI.

`tests/test_model.py`: PyTest test of mode. Gets run with CI.

`test_api.py`: Script that tests the live API. Must be run manually.

`screenshots/*.png`: Images showing various parts of the pipeline working.

`slice_output.txt`: Example output of ML model performance on slices.


## Creating the Pipeline

The steps I took to create this pipeline were roughly as follows.

1.

2.

3.




## Rubric Points

* 

