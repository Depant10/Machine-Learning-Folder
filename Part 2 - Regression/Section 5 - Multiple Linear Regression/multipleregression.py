# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 03:46:32 2020

@author: devang
"""
#data processing
#impporting the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as asd

# Importing the dataset
dataset = asd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values
#encoding the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
#Dummyvariable trap
X = X[:,1: ]
#splitting into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#Importing the regressor on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
#predicting the profit
y_pred = regressor.predict(X_test)
#importing the value of Bo, axis = 1 means add into column
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis= 1)
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
reg_OLS = sm.OLS(endog= Y,exog= X_opt).fit()
reg_OLS.summary()
X_opt = X[:,[0, 1, 3, 4, 5]]
reg_OLS = sm.OLS(endog= Y,exog= X_opt).fit()
reg_OLS.summary()
X_opt = X[:,[0, 3, 4, 5]]
reg_OLS = sm.OLS(endog= Y,exog= X_opt).fit()
reg_OLS.summary()
X_opt = X[:,[0, 3, 5]]
reg_OLS = sm.OLS(endog= Y,exog= X_opt).fit()
reg_OLS.summary()
X_opt = X[:,[0, 3,]]
reg_OLS = sm.OLS(endog= Y,exog= X_opt).fit()
reg_OLS.summary()




