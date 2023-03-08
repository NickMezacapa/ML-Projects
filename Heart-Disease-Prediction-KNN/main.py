import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def main():
    # import the data
    data_df = pd.read_csv('./data/heart_disease_cleveland_uci.csv')

    # use seaborn to visualize distribution of target variable with a countplot
    sns.countplot(data_df['target'])

    # define independent and dependent variables
    x = data_df.iloc[:, 0:13].values
    y = data_df['target'].values

    # split the data into training and testing sets (75/25 split)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # scale the data using StandardScaler
    st_x = StandardScaler()
    x_train = st_x.fit_transform(x_train)
    x_test = st_x.transform(x_test)

    # initialize the KNN classifier

    # initialize an empty list to store error rate for different K values (misclassified samples in test set)
    error = []
    # calculate error for K values between 1 and 30
    for i in range(1, 30):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        error.append(np.mean(pred_i != y_test))

    # plot the error rate for different K values
    plt.figure(figsize = (12, 6))
    plt.plot(range(1, 30), error, color='red', linestyle='dashed', marker='o',
                markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    # print the minimum error and the K value at which it occurs
    print('Minimum error:-', min(error), 'at K =', error.index(min(error)) + 1)


    """
    KNN is a simple but effective algorithm for classification tasks,
    which uses the k closest neighbors in the training set to predict 
    the class of a new sample.
    """

    # initialize the KNN classifier with the optimal K value, then fit to training set
    classifier = KNeighborsClassifier(n_neighbors = 7)
    classifier.fit(x_train, y_train)

    # predict the test set results
    y_pred = classifier.predict(x_test)

    # evaluate the model using a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # calculate & print the accuracy score
    print('Accuracy:-', accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    main()
