# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 05:55:36 2020

@author: devan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset= pd.read_csv('Mall_Customers.csv')
X= dataset.iloc[:, [3,4]].values
#using denogram to find optimal number of cluster
import scipy.cluster.hierarchy as sch
dendrogram= sch.dendrogram(sch.linkage(X, method= 'ward'))
plt.title('denogram')
plt.xlabel('customers')
plt.ylabel('spend')
plt.show()
#fitting hierarchical clustering into dendogram
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters= 5, affinity= 'euclidean', linkage= 'ward')
y_pred= hc.fit_predict(X)
#visualising the cluster
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0,1], s= 100, c= 'blue', label= 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1,1], s= 100, c= 'red', label= 'Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2,1], s= 100, c= 'cyan', label= 'Cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3,1], s= 100, c= 'green', label= 'Cluster 4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4,1], s= 100, c= 'magenta', label= 'Cluster 51')
plt.title('Cluster of clients')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()