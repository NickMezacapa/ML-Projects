from csv import reader
from math import sqrt, exp
from random import randrange
from random import seed
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np


# creating a function to read in a CSV file and convert it to a list of lists
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# creating a function to convert columns in dataset from a string to a float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# creating a function to compute the minimum and maximum values for each column in dataset
# returns each column's minimum and maximum values in a list of tuples
def dataset_minmax(dataset):
    # zip(*dataset) transposes dataset matrix so that each column is treated as a single list
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# creating a function to normalize the dataset
# scales each value to be between 0 and 1 based on the min and max values
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# creating a function to split the dataset into k folds for cross validation
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# creating a function to calculate accuracy of predicted values
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    """
    Evaluates the algorithm using k-fold cross-validation.

    Splits the dataset into folds, then for each fold, trains the model on the
    remaining folds and tests the model on the fold that was held back.

    Parameters:
    - dataset (list): The dataset to evaluate.
    - algorithm (function): The algorithm to evaluate.
    - n_folds (int): The number of folds to use for cross-validation.
    - *args: Additional arguments to pass to the algorithm.

    Returns:
    - scores (list): A list of accuracy scores for each fold.
    """
    folds = cross_validation_split(dataset, n_folds)
    scores = {'accuracy': [], 'auc': [], 'f1': [], 'precision': [], 'recall': []}
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted_probs = algorithm(train_set, test_set, *args)
        predicted = [round(p) for p in predicted_probs]
        actual = [row[-1] for row in fold]
        # will calculate accuracy, auc, f1, precision, and recall, then create a histogram of the predicted probabilities
        accuracy = accuracy_metric(actual, predicted)
        auc = roc_auc_score(actual, predicted_probs)
        f1 = f1_score(actual, predicted)
        precision = precision_score(actual, predicted)
        recall = recall_score(actual, predicted)
        scores['accuracy'].append(accuracy)
        scores['auc'].append(auc)
        scores['f1'].append(f1)
        scores['precision'].append(precision)
        scores['recall'].append(recall)
        plt.hist(predicted_probs, bins=10)
        plt.title('Probability Scores Histogram')
        plt.xlabel('Probability Score')
        plt.ylabel('Frequency')
        plt.show()
    return scores


def predict(row, coefficients):
    """
    Predicts the class label for a row of data based on the logistic regression coefficients.

    Predicts the class label by computing the dot product of the row and the coefficients,
    and applying the sigmoid function to the result. If the result is greater than or equal
    to a threshold of 0.5, the function returns 1, otherwise it returns 0.

    Parameters:
    - row (list): A list of data values for a row of data.
    - coefficients (list): A list of coefficients for each column in the row.

    Returns:
    - yhat (float): The predicted class label (0 or 1).
    """
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat)) # sigmoid function


def coefficients_sgd(train, l_rate, n_epochs):
    """
    Calculates the coefficients of the logistic regression model using stochastic gradient descent.

    Uses stochastic gradient descent (SGD) to update the coefficients iteratively based
    on the error of the predictions compared to the actual class labels. For each epoch, 
    it loops through each row in the training dataset, and updates the coefficients based
    on the prediction error for that row. The learning rate determines the size of the 
    coefficient update. The number of epochs determines the number of iterations.

    Parameters:
    - train (list): a list of lists representing training dataset.
    - l_rate (float): the learning rate.
    - n_epochs (int): the number of training epochs.

    Returns:
    - coefficients (list): a list of coefficients for each column in the dataset.
    """
    coefficients = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epochs):
        for row in train:
            yhat = predict(row, coefficients)
            error = row[-1] - yhat
            coefficients[0] = coefficients[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row)-1):
                coefficients[i + 1] = coefficients[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
    return coefficients


def logistic_regression(train, test, l_rate, n_epoch):
    """
    Makes predictions using logistic regression.

    Uses the coefficients_sgd function to calculate the coefficients of the logistic
    regression model, then uses those coefficients to make predictions on the test set.

    Parameters:
    - train (list): a list of lists representing training dataset.
    - test (list): a list of lists representing test dataset.
    - l_rate (float): the learning rate.
    - n_epochs (int): the number of training epochs.

    Returns:
    - predictions (list): a list of predicted class labels.
    """
    predictions = list()
    coefficients = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, coefficients)
        yhat = round(yhat)
        predictions.append(yhat)
    return(predictions)


# loading the dataset
seed(1) # setting the seed for reproducibility
filename = './data/diabetes.csv'
dataset = load_csv(filename)


# loop through each column and convert string values to float
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)


minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)


# evaluating the algorithm
n_folds = 5
l_rate = 0.1
n_epoch = 100
scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)


# printing the scores
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
