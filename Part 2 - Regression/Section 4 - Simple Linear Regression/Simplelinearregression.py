 # -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:33:55 2020

@author: devang
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as asd
#importing the dataset
dataset =  asd.read_csv('Salary_Data.csv')
X= dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1].values
#splitting data into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state= 0)
#Training simple regression on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
#predicting the result
Y_pred = regressor.predict(X_test)
#Visualising the training set
plt.scatter(X_train,Y_train,color= 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title('salary vs experience train data')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show
#visualising the test results
plt.scatter(X_test,Y_test,color= 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title('salary vs experience testdata')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show