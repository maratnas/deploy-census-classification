# Model Card

## Model Details

Creator: Karl J. Obermeyer (karl.obermeyer@gmail.com)  
Date created: 2022:10:31  
Version: 1.0.0-alpha  
Model type: random forest classifier  
Inputs: demographic features about a person  
Output: inferred most likely salary class of person ("<=50K" or ">50K" US dollars)  
Licese: MIT


## Intended Use

This model infers which salary class a person most likely falls into based on
demographic features found in census data. The model is only intended as a proof
of concept for educational purposes (Udacity ML DevOps Nanodegree project #3).
Before being put into production, the model should be further examined for
biases as described in the "Ethical Considerations" section below.


## Training Data

Data, which originally was [published by the US Census
Bureau](https://archive.ics.uci.edu/ml/datasets/census+income), was downloaded
from Udacity at
`https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/data/census.csv`. The full data frame consisted of 32561 rows and the following 15 columns:

```
age
workclass
fnlgt
education
education-num
marital-status
occupation
relationship
race
sex
capital-gain
capital-loss
hours-per-week
native-country
salary
```
The full data frame was randomly shuffled and split into training (80%) and test
(20%) examples. Categorical data was one-hot encoded and labels ("<=50K" or
">50K") were passed through a binarizer


## Evaluation Data

Testing was performed on a random 20% subset of the full dataset as described
above.


## Metrics

Precision: 0.7306  
Recall: 0.5997  
F1: 0.6587  


## Ethical Considerations

The data looked roughly balanced with respect to many features in a cursory
inspection. One notable exception, which is not surprising from the US Census
Bureau, is that most of the data is from people in the US. The data should be
studied more closely for fair representation before the model is used for any
application of consequence.


## Caveats and Recommendations

* Refrain from using this model for applications of consequence until the data has been more carefully examined for fairness.
* There are only 2 salary categories currently ("<=50K" and ">50K"). Consider refining.
