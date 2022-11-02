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

`tests/test_api.py`: PyTest of API running locally.

`query_live_api.py`: Script that queries the live API and prints the
responses.

`screenshots/*.png`: Images showing various parts of the pipeline working.

`slice_output.txt`: Example output of ML model performance on slices.


## Creating the Pipeline

This is a high-level outline of the steps I took to create this pipeline.

1.

2.

3.
.
.
.


## Rubric Points

* GitHub Actions are specified in `.github/workflows/python-app.yml`.

* Pytests are specified in `tests/` and they pass.

* Flake8 passes.

* `screenshots/continuous_integration.png` shows CI passing.

* The modules `ml/*.py` implement functions for
  - train, save, and load model and encoders,
  - inference,
  - metrics evaluation.

* The script `train_model.py` performs a train-test split, preprocesses data,
  trains the model, and pickles the model together with encoders/decoders as an
  `AugmentedModel` type.

* More than 3 unit tests are implemented in `tests/test_model.py`.

* The function `compute_model_metrics_for_all_slices_of_a_feature` in
  `ml/evaluation.py` computes slice metrics, and the script
  `compute_slice_metrics.py` uses it to print metrics for all slices of a
  particular feature. The file `slice_output.txt` shows an example of the
  model's performance on gender slices.

* The model card `model_card.md` is complete.

* A REST API is implemented with FastAPI in `main.py`.
  -GET returns a greeting.
  -POST returns an inferred salary class.
  -Type hints are used.
  -Uses Pydantic model `CensusFeatures` from `census_features.py`.
  -Screenshot `screenshots/example.png` shows example `CensusFeatures` instance
  (in small text at the bottom).

* The pytest `tests/test_api.py` locally tests the API responses of the cases of
  GET, low-salary POST, and high-salary POST.

* App continuously deploys from GitHub to Heroku and
  `screenshots/continuous_deployment.png` shows it.

* `live_get.png` shows API GET response in browser.

* The script `query_live_api.py` uses `requests` to query the live API for the
cases of GET, low-salary POST, and high-salary POST. Responses are printed to
the stdout. `live_post.png` shows the results printed in the terminal. Also
`live_post-heroku_logs.png` shows the logs of what is happening on the server.
