# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:38:58 2019

@author: Rupesh S
"""

import pandas as pd

data=pd.read_csv("D:\\logistic_regression.csv")

x=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

accuracy=(cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,0]+cm[0,0]+cm[1,1])

#Find the class probability
classifier.predict_proba(x_test)