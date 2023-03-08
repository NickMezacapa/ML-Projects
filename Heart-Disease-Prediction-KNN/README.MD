# Heart Disease Prediction using K-Nearest Neighbors Algorithm

This project aims to predict whether a patient has heart disease based on various medical attributes. The dataset used in this project is from the UCI Machine Learning Repository, which has been re-processed and cross-checked with the original data to correct some errors. The dataset contains 297 instances of patients with 14 attributes each. The attributes are described below.
<br>

## Dataset Attributes
The dataset consists of the following attributes and includes one label:

1. age: age of the patient in years
2. sex: gender of the patient (1 = male; 0 = female)
3. cp: chest pain type (0 = typical angina; 1 = atypical angina; 2 = non-anginal pain; 3 = asymptomatic)
4. trestbps: resting blood pressure in mm Hg on admission to the hospital
5. chol: serum cholestoral in mg/dl
6. fbs: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. restecg: resting electrocardiographic results (0 = normal; 1 = having ST-T wave abnormality; 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
8. thalach: maximum heart rate achieved
9. exang: exercise induced angina (1 = yes; 0 = no)
10. oldpeak: ST depression induced by exercise relative to rest
11. slope: the slope of the peak exercise ST segment (0 = upsloping; 1 = flat; 2 = downsloping)
12. ca: number of major vessels colored by flourosopy (0-3)
13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
14. condition: 0 = no disease; 1 = disease (label)
<br>

## Data Preprocessing
Before training the model, the dataset was preprocessed to handle missing values and normalize the data. The missing values were imputed with the mean value of the respective attribute. The data was then normalized using MinMaxScaler from the scikit-learn library.
<br>

## Model
The model used in this project is a K-Nearest Neighbors classifier from the scikit-learn library. The model was trained on 75% of the dataset and tested on the remaining 25%. The accuracy achieved by the model was 86.8%. Improvements can be made by tuning the hyperparameters of the K-Nearest Neighbors classifier.
