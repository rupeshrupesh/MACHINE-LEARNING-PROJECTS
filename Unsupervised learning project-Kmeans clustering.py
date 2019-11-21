# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:55:32 2019

@author: Rupesh S
"""

import pandas as pd

data=pd.read_csv("D:\\Mall_Customers.csv")
data.columns

x=data.iloc[:,[3,4]].values

from sklearn.cluster import KMeans

WCSS=[]

for i in range(1,11):
    k=KMeans(i,init='k-means++', random_state=123)
    k.fit(x)
    WCSS.append(k.inertia_)
    
import matplotlib.pyplot as plt

plt.plot(range(1,11), WCSS)
plt.show()

km=KMeans(n_clusters=5, init='k-means++', random_state=123)

y_pred=km.fit_predict(x)

plt.scatter(x[y_pred==0,0], x[y_pred==0,1], s=100, c='red',label='Cluster1')
plt.scatter(x[y_pred==1,0], x[y_pred==1,1], s=100, c='blue',label='Cluster1')
plt.scatter(x[y_pred==2,0], x[y_pred==2,1], s=100, c='cyan',label='Cluster1')
plt.scatter(x[y_pred==3,0], x[y_pred==3,1], s=100, c='black',label='Cluster1')
plt.scatter(x[y_pred==4,0], x[y_pred==4,1], s=100, c='green',label='Cluster1')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=100,c='yellow',label='centroids')
plt.legend()
plt.show()