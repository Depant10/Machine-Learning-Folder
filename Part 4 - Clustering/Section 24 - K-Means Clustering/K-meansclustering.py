# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 03:42:51 2020

@author: devang
"""
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset= pd.read_csv('Mall_Customers.csv')
X= dataset.iloc[:, [3,4]].values
#USING THE METHODTOCALCULATE THE OPTIMAL NUMBER OF CLUSTERS
from sklearn.cluster import KMeans
wcss= []
for i in range(1,11):
   kmeans= KMeans(n_clusters=i , init= 'k-means++', max_iter= 300, n_init= 10, random_state= 0)
   kmeans.fit(X)
   wcss.append(kmeans.inertia_)
#plotting the graph of clusters
plt.plot(range(1, 11), wcss)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show
#applying kmeansin the algorithm
kmeans= KMeans(n_clusters=i , init= 'k-means++', max_iter= 300, n_init= 10, random_state= 0)
y_pred= kmeans.fit_predict(X)
#visualising the cluster
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0,1], s= 100, c= 'blue', label= 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1,1], s= 100, c= 'red', label= 'Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2,1], s= 100, c= 'cyan', label= 'Cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3,1], s= 100, c= 'green', label= 'Cluster 4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4,1], s= 100, c= 'magenta', label= 'Cluster 51')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s= 300, c= 'yellow', label= 'centeroids')
plt.title('Cluster of clients')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()
