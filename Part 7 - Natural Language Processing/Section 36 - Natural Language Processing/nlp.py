
"""
Created on Thu Apr 30 05:10:23 2020

@author: devan
"""
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset= pd.read_csv('Restaurant_Reviews.tsv', delimiter= '\t', quoting= 3, engine= 'python')
#cleaning and stemming the text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
corpus= []
for i in range(0, 1000):
    review= re.sub('[^a-zA-z]', ' ' ,dataset['Review'][i])
    review= review.lower()
    review= review.split()
    ps= PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review= ' '.join(review)
    corpus.append(review)
#creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features= 1500)
X= cv.fit_transform(corpus).toarray()
y= dataset.iloc[:, 1].values
#splitting into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

"""#feature scaling
from sklearn.preprocessing  import StandardScaler
sc_X= StandardScaler()
sc_y= StandardScaler()
X_test= sc_X.fit_transform(X_test)
X_train= sc_y.fit_transform(X_train)"""
#create your classifier
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(X_train, y_train)
#predicting the results
y_pred= classifier.predict(X_train)
#maing conclusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_train, y_pred)

