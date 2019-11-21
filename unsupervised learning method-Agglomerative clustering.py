# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:17:56 2019

@author: Rupesh S
"""

import pandas as pd

df=pd.read_csv("D:\\Mall_Customers.csv")

x=df.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

dendrogram=sch.dendrogram(sch.linkage(x, method='ward'))

plt.xlabel('customers')
plt.ylabel('euclidean distane')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5, affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)

plt.scatter(x[y_hc==0,0], x[y_hc==0,1], s=100, c='red',label='Cluster1')
plt.scatter(x[y_hc==1,0], x[y_hc==1,1], s=100, c='blue',label='Cluster1')
plt.scatter(x[y_hc==2,0], x[y_hc==2,1], s=100, c='cyan',label='Cluster1')
plt.scatter(x[y_hc==3,0], x[y_hc==3,1], s=100, c='black',label='Cluster1')
plt.scatter(x[y_hc==4,0], x[y_hc==4,1], s=100, c='green',label='Cluster1')
plt.scatter()