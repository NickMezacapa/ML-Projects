import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main():
    # import the data
    data_df = pd.read_csv('./data/power_plant.csv')

    # define the independent and dependent variables
    x = data_df.drop(['PE'], axis=1).values
    y = data_df['PE'].values

    # split the data into training and testing sets (70% train / 30% test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # train the model on the training set
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # make predictions on the test set
    y_pred = regressor.predict(x_test)

    # evaluate the model
    r2_score(y_test, y_pred)

    # plot the results
    plt.figure(figsize=(15, 10))
    plt.scatter(y_test, y_pred) # plot the actual vs predicted values
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Net Hourly Electrical Energy Output')

    # print the actual and predicted values, as well as residuals
    y_pred_df = pd.DataFrame({'Actual Value': y_test, 'Predicted Value': y_pred, 'Difference': y_test - y_pred})
    y_pred_df[0:20]
    print(y_pred_df)

if __name__ == '__main__':
    main()

