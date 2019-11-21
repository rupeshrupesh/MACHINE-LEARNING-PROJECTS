# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:37:23 2019

@author: Rupesh S
"""

import pandas as pd

data=pd.read_csv("D:\\Churn_Modelling.csv")

x=data.iloc[:,3:13].values
y=data.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
x[:,1]=label.fit_transform(x[:,1])
x[:,2]=label.fit_transform(x[:,2])

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features=[1])

x=ohe.fit_transform(x).toarray()
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=0)

#Random forest algorithms
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=50,criterion='entropy', random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])

#Cross validation method using k-fold.To improve accuracy
from sklearn.model_selection import cross_val_score
cv_acc=cross_val_score(classifier, x_train, y_train, cv=10)
cv_acc.mean()
cv_acc.std()

#GridsearchCv mostly works in SVM. to finding the best estimator
from sklearn.model_selection import GridSearchCV

param_grid={'bootstrap':[True],'n_estimators':[10,20,50,100]}
classifier_grid= RandomForestClassifier()
grid_search=GridSearchCV(classifier_grid, param_grid, cv=10, n_jobs=-1)
grid_search.fit(x_train,y_train)
grid_search.best_params_
grid_search.best_estimator_


#XGBOOST 
from xgboost.sklearn import XGBClassifier
classifier1=XGBClassifier()
classifier1.fit(x_train,y_train)
y_pred=classifier1.predict(x_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
xgb_accuracy1=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])

print(accuracy)
print(xgb_accuracy1)

