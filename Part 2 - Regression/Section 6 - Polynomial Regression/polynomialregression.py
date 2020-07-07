# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 02:51:21 2020

@author: devang
"""
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
#splitting into training and test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
#fitting into linnear regression
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X,y)
#fitting into polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
#visualising the polynomial regression results
X_grid= np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color= 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color= 'blue')
plt.title('truth or bluff(polynomial regression')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()
#predicting using polynomial regression
y_pred= lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
