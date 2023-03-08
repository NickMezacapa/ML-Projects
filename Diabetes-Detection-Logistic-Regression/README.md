# Predicting Type 2 Diabetes Using Logistic Regression

This machine learning project aims to predict the probability of whether an individual is likely to develop type 2 diabetes or not. The dataset used in this project is the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), which is available on Kaggle and originally made public by the [National Institute of Diabetes and Digestive and Kidney Diseases](https://www.niddk.nih.gov/).

A logistic regression implementation using stochastic gradient descent is used to predict the probability of an individual developing type 2 diabetes based on several clinical features such as age, BMI, blood pressure, and glucose level. The model is trained on a dataset containing data on 768 patients, of which 268 have already been diagnosed with diabetes.<br>
<br>

## Introduction

[Diabetes mellitus](https://www.cdc.gov/diabetes/basics/diabetes.html) is one of the most common human diseases worldwide. It is a chronic health condition that affects the way the body processes blood sugar (glucose). The body needs insulin to convert glucose into energy, and in individuals with diabetes, the body either doesn't produce enough insulin or can't effectively use the insulin it produces. This causes glucose to build up in the blood, which can lead to serious health complications. It is responsible for considerable morbidity, mortality, and healthcare costs. Among diabetes patients, type 2 diabetes accounts for ~90-95% of cases. <br>

The main risk factos for type 2 diabetes include (but are not limited to) obesity, body mass index, physical inactivity, family history, age, ethnicity, and history of gestational diabetes.<br>

A timely diagnosis of diabetes is needed for patients to take appropriate measures to prevent the onset of complications. Prediction models can be used to screen individuals with an increased risk of developing diabetes to help decide the best clinical management for patients. The prediction performance and validity of these models can vary significantly depending on the input variables that are used. <br>
<br>

## Dataset

The dataset used in this project is the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), which is available on Kaggle and originally made public by the [National Institute of Diabetes and Digestive and Kidney Diseases](https://www.niddk.nih.gov/). The dataset contains data on 768 patients, of which 268 have already been diagnosed with diabetes. The dataset contains 8 features, which are described below:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**: Class variable (0 or 1) 268 of 768 are 1, the others are 0<br>

The dataset contains several missing values, which are represented by zeros. These values are replaced with the mean of the respective feature. [K-fold cross-validation](https://towardsdatascience.com/k-fold-cross-validation-explained-in-plain-english-659e33c0bc0) is used to evaluate the model performance. The dataset is split into 5 folds, and the model is trained on 4 folds and evaluated on the remaining fold.<br>
<br>

## Model

[Logistic regression](https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148) is a statistical method used to analyze a dataset with one or more independent variables that determine an outcome. It is commonly used in machine learning to predict binary outcomes, such as whether a patient has a disease or not (i.e. type 2 diabetes / no type 2 diabetes).<br>

[Stochastic gradient descent](https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31) (SGD) is a widely used optimization algorithm in machine learning that iteratively updates the model parameters based on the gradients of the loss function computed on a subset of the training data. Compared to batch gradient descent, which updates the model parameters after computing the gradients on the entire training set, SGD is more computationally efficient and can scale to very large datasets.<br>

In this project, a logistic regression implementation using stochastic gradient descent is used to predict the probability of an individual developing type 2 diabetes based on several clinical and demographic features. By using SGD, the model can be trained efficiently on a large dataset and achieve good prediction accuracy with a relatively small number of epochs. The k-fold cross-validation technique is used to evaluate the performance of the model and prevent overfitting.<br>
<br>


## Results

The logistic regression model using stochastic gradient descent was evaluated using k-fold cross-validation with k=5. The model achieved an average [accuracy](https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9) of 77.1% over the 5 folds, with a [standard deviation](https://deepai.org/machine-learning-glossary-and-terms/standard-deviation) of 2.5%. The model achieved an average [AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) of 0.84 over the 5 folds, with a standard deviation of 0.03. The model achieved an average [F1](https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9) score of 0.69 over the 5 folds, with a standard deviation of 0.04. The model achieved an average [precision](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) of 0.66 over the 5 folds, with a standard deviation of 0.04. The model achieved an average [recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) of 0.73 over the 5 folds, with a standard deviation of 0.04.<br>

Overall, these metrics suggest that the logistic regression model using stochastic gradient descent is reasonably effective for predicting the probability of an individual developing type 2 diabetes based on several clinical and demographic features.<br>
<br>


## Project Dependencies
- Python 3.x
- Numpy
- Matplotlib
- Scikit-learn

These dependencies are only used to find accuracy metrics and plot the ROC curve. The model can be trained and evaluated without these dependencies.<br>
<br>

## Installation

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/NickMezacapa/Diabetes-Detection-Logistic-Regression.git
cd Diabetes-Detection-Logistic-Regression
```
<br>

## Usage

Execute the following command to train and evaluate the model:

```bash
python main.py
```
