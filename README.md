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

`census_features.py`: Defines `CensusFeatures` type, derived from
  `pydantic.BaseModel`, for posting to the API. This is by the app to ingest
  POSTs in the `infer` function of `main.py`.

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


## Setting up the Pipeline

This is a high-level outline of the main steps I took to set up this pipeline.

* Create GitHub and Heroku accounts if you do not already have them.

* Locally install conda, git, and dvc if not already present.

* Create and activate a conda environment for local development.

```
$ conda create -n deploy-census-classification "python=3.7" \
    requests \
    pytest \
    flake8 \
    numpy \
    scipy \
    matplotlib \
    scikit-learn \
    pandas \
    pandas-profiling \
    jupyter \
    jupyterlab \
    fastapi \
    uvicorn \
    dvc-s3 \
    dvc-gdrive \
    -c conda-forge
$ conda activate deploy-census-classification
```

* Create, resp. fork the repo `deploy-census-classification` on GitHub.

* Clone repo to local machine for development.
```
git clone git@github.com:maratnas/deploy-census-classification.git
```

* Add file `requirements.txt` that enclodes python dependencies so that GitHub
and Heroku can use pip rather than conda because pip is much faster.

* Set up a "Python Package/Application" pre-made GitHub Action in the repo.
This should run pytest and flake8 on push, and require both to pass. Set the
Python version to be the same as that used for development.

* From Udacity ND menu, select "OPEN CLOUD GATEWAY". Credentials
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` appear, but ignore these. We
will instead create an IAM user and use its credentials instead.

* [Install AWS CLI
  tool](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
  and confirm.
```
$ aws --version
```

* [Create s3
bucket](https://s3.console.aws.amazon.com/s3/buckets?region=us-west-2) named
`census-classification-0` with default options: Services -> Create Bucket.

* [Create IAM
user](<https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console>)
named `census-classification-0-user`. Note the values of `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`. Users -> Add user Programmatic access. Attach Existing
Policies Directly (don't create group). Search for and select
  `AmazonS3FullAccess` to give S3 bucket access. Skip tags. Create user.
Configure AWS CLI to use Access key ID and Secret Access key.
```
$ aws configure
AWS Access Key ID [None]: ...
AWS Secret Access Key [None]: ...
Default region
name [None]: us-west-2 Default output format [None]: json
```

* Initialize DVC
```
$ dvc init
```

* Add S3 DVC remote as default.
```
$ dvc remote add s3-store -d s3://census-classification-0
$ git add .dvc/config
$ git commit -m "Configure S3 DVC remote."
```

* [Connect S3 to GitHub
Actions](https://knowledge.udacity.com/questions/669541). -[Add AWS
credentials](https://github.com/marketplace/actions/configure-aws-credentials-action-for-github-actions)
and [DVC setup](https://github.com/iterative/setup-dvc) to Action.

* Download and commit raw data from starter repo.
```
$ wget -P ./data/ \
  <https://github.com/udacity/nd0821-c3-starter-code/raw/master/starter/data/census.csv>
$ dvc add data/census.csv
$ git add data/census.csv.dvc data/.gitignore
$ git commit -m "Add raw data."
```

* Open `census.csv` with a text editor, remove all spaces, save result
  as `census-clean.csv`, and add to DVC.
```
$ dvc add data/census-clean.csv
$ git add data/census-clean.csv.dvc data/.gitignore
$ git commit -m "Add cleaned data."
```

* Implement the ML functionality, train a model and serialize the model and a
test data frame to `models/model.pkl` and `data/data_frame_test.pkl`,
respectively.

* Add serialized model and test dataset to DVC.
```
$ dvc add models/model.pkl
$ git add models/model.pkl.dvc models/.gitignore
$ git commit -m "Add model artifact."
$ dvc add data/data_frame_test.pkl
$ git add data/data_frame_test.pkl.dvc data/.gitignore
$ git commit -m "Add a serialized test data frame."
```

* Push tracked data to S3 remote.
```
$ dvc push
```

* Implement API app.

* Add IAM user credentials to [GitHub repo secrets](https://github.com/maratnas/deploy-census-classification/settings/secrets/actions).

* Push to GitHub to see if triggered tests pass,
```
$ git push
```

* Create Heroku app named `census-classification-a4a401cd`.

* Add `apt` to Heroku VM buildpack so that it can install DVC.
```
$ heroku buildpacks:add --index 1 heroku-community/apt --app census-classification-a4a401cd
$ git push heroku main  # Create new release using the buildback.
```

* [Add `heroku/python` buildpack](https://dashboard.heroku.com/apps/census-classification-a4a401cd/settings).

* Add `Procfile` for Heroku.
``` Procfile
web: uvicorn main:app --host=0.0.0.0 --port=${PORT:-5000}
```

* Add `Aptfile` for Heroku.
``` Aptfile
https://github.com/iterative/dvc/releases/download/2.10.2/dvc_2.10.2_amd64.deb
```

* [Set AWS credentials as "Config Vars" in
  Heroku](https://dashboard.heroku.com/apps/census-classification-a4a401cd/settings).

* Deploy app to Heroku using GitHub repo with CD enabled.
  - Enable autodeploy contingent on CI passing.
```
$ git push
```


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
  - GET returns a greeting.
  - POST returns an inferred salary class.
  - Type hints are used.
  - The `infer` function ingests POSTs with a Pydantic model `CensusFeatures`
    derived from `pydantic.BaseModel` in `census_features.py`.
  - Screenshot `screenshots/example.png` shows example `CensusFeatures` instance
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
