import pandas as pd
import numpy as np
import sklearn.tree as tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def main():
    # import the data
    data_df = pd.read_csv('./data/drug_response_data.csv', delimiter=',')

    """
    Some features in this dataset are categorical, such as 'Sex' or 'BP'.
    Sklearn does not handle categorical variables. We can convert
    these features to numerical values using `pandas.get_dummies()`. 
    'X' will be defined as the feature matrix, and 'y' will be defined as
    the response vector.
    """

    # define X and y, preprocess the data
    X = data_df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

    le_sex = preprocessing.LabelEncoder()
    le_sex.fit(['F', 'M']) # Female and Male
    X[:, 1] = le_sex.transform(X[:, 1]) # convert to numerical values

    le_BP = preprocessing.LabelEncoder()
    le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
    X[:, 2] = le_BP.transform(X[:, 2])

    le_Chol = preprocessing.LabelEncoder()
    le_Chol.fit(['NORMAL', 'HIGH'])
    X[:, 3] = le_Chol.transform(X[:, 3])

    y = data_df['Drug'].values

    # setting up the decision tree
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
    print('Shape of X_train: {}'.format(X_train.shape), 'Shape of y_train: '.format(y_train.shape))
    print('Shape of X_test: {}'.format(X_test.shape), 'Shape of y_test: '.format(y_test.shape))

    # creating instance of the DecisionTreeClassifier 
    drug_tree = DecisionTreeClassifier(criterion='entropy', max_depth=4)

    # fitting the data with training feature matrix and response vector
    drug_tree.fit(X_train, y_train)

    # making predictions on the testing set
    prediction_tree = drug_tree.predict(X_test)
    print(prediction_tree [0:5])
    print(y_test [0:5])

    """
    Evaluating the accuracy of the model using metrics from sklearn.
    Accuracy classification score computes subset accuracy: the set of
    labels predicted for a sample must exactly match the corresponding set 
    of labels in y_true. If the entire set of predicted labels for a sample
    strictly matches with the true set of labels, then the subset accuracy
    is 1.0; otherwise it is 0.0.
    """
    print('DecisionTree Accuracy: ', metrics.accuracy_score(y_test, prediction_tree))

    # visualizing the tree
    tree.plot_tree(drug_tree)
    plt.show()


if __name__ == '__main__':
    main()
