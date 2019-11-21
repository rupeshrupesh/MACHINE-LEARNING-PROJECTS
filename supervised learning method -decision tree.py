# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:36:54 2019

@author: Rupesh S
"""

import pandas as pd

data=pd.read_csv("D:\\logistic_regression.csv")

x=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy')
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

import pydotplus
from sklearn.tree import export_graphviz

dot=export_graphviz(classifier, out_file=None, filled=True, rounded=True)
graph=pydotplus.graph_from_dot_data(dot)
graph.write_png('sample.png')


