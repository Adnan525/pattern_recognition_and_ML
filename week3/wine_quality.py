# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 14:07:53 2020

@author: s434074
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

#split the data to train/test dataset
from sklearn.model_selection import train_test_split

#import the linear regression
from sklearn.linear_model import LinearRegression

#read the wine quality data
wine_red_dataset = pd.read_csv("winequality-red.csv", sep=';')
wine_white_dataset = pd.read_csv("winequality-white.csv", sep=';')

wine_dataset = pd.concat([wine_red_dataset, wine_white_dataset])

X = wine_dataset.drop(['quality'], axis = 1)

#X = wine_dataset[['fixed acidity']]
#correct petal width values
Y = wine_dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.25, 
                                                    random_state= 1)

lre = LinearRegression()

#train the train data
lre.fit(X_train, y_train)

#predict on the test data
pred = lre.predict(X_test)

print(lre.intercept_)
print(lre.coef_)

#Evaluate the prediction
print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
print('Mean Squared Error:', mean_squared_error(y_test, pred))
print('Mean Root Squared Error:', np.sqrt(mean_squared_error(y_test, pred)))
#print(abs(Y - Y_pred))

plt.scatter(X_test[['citric acid']], y_test, color = "red", 
            label = "Actual Wine Quality")
plt.scatter(X_test[['citric acid']], pred, color = "green", 
            label = "Predicted Wine Quality")
plt.legend()
plt.xlabel('Citric Acid')