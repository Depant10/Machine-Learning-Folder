#data processing
#impporting the libraries

import numpy as np #same as math.h in c
import matplotlib.pyplot as plt #everytime we plot, we use matplotlib
import pandas as asd

# Importing the dataset
dataset = asd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values
#handling the missing data
from sklearn.impute import simpleimputer
imputer = simpleimputer(missing_values = np.nan, strategy= 'mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
#splitting into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

