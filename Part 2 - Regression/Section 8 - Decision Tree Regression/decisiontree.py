# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 05:13:37 2020

@author: devang
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries (1).csv')
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values
#reshaping the vallues
X= X.reshape(-1, 1)
y= y.reshape(-1, 1)


#feature scaling
"""from sklearn.preprocessing  import StandardScaler
sc_X= StandardScaler()
sc_y= StandardScaler()
X= sc_X.fit_transform(X)
y= sc_y.fit_transform(y)
#splitting into training and test set"""
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

#fitting decisiontree into dataset
from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor()
regressor.fit(X, y)


#predicting the values
Y_pred = regressor.predict(np.array([[6.5]]))
#visualising the polynomial regression results
X_grid= np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color= 'red')
plt.plot(X_grid, regressor.predict(X_grid), color= 'blue')
plt.title('truth or bluff(DecisionTree')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()
