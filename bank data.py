# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 01:41:50 2019

@author: Rupesh S
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_excel("D:\\project1.xlsx")
df.count()
df.columns
df.isnull().sum()

df['age'].plot(kind='hist', bins=30, normed=True)
df['job'].values
df['job'].value_counts()
df['marital'].value_counts()
df['education'].value_counts()
df[['default','housing','loan']].values
df['month'].value_counts()
df['day_of_week'].value_counts()
df['campaign'].value_counts()
df['pdays'].value_counts()
df['previous'].value_counts()
df['poutcome'].value_counts()
s1=df.corr()
df=df.drop(['cons.conf.idx','euribor3m','nr.employed'], axis='columns')
df=df.drop(['emp.var.rate','cons.price.idx'],axis=1)

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
df['y']=label_encoder.fit_transform(df['y'])
df.corr()

df.columns
df=pd.get_dummies(df,columns=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome'])


df=df.drop(columns=['job_no','marital_no','education_no','default_no','housing_no','loan_no','month_aug','day_of_week_fri','poutcome_nonexistent'],axis=1)

#DATA
x=df.drop('y',axis=1)
y=df['y'].values

#Dimensionality Reduction
from sklearn.decomposition import PCA
pca=PCA()
x=pca.fit_transform(x)

pca.components_[0] #eigen values

pca.explained_variance_ratio_
res=pca.explained_variance_ratio_*100

import numpy as np
res=np.cumsum(pca.explained_variance_ratio_*100)

scores=pd.Series(pca.components_[0])
scores1=scores.abs().sort_values(ascending=False)
var=pca.components_[0]
plt.bar(x=range(1,len(var)+1),height=res)
plt.show()

s1=scores1.head(3)
s2=df.iloc[:,[0,1,2,3]]
s2['y']=df['y']

import statsmodels.formula.api as sm

def vif_cal(input_data,dependent_col):
    x_var=input_data.drop([dependent_col], axis=1)
    xvar_names= x_var.columns
    for i in range(0, len(xvar_names)):
        y=x_var[xvar_names[i]]
        x=x_var[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols('y~x',x_var).fit().rsquared
        vif=round(1/(1-rsq),2)
        print(xvar_names[i],"VIF:",vif)
vif_cal(s2,'y')

x=s2.iloc[:,[0,1,2,3]]
y=s2.iloc[:,4].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=123)

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

import seaborn as sns
sns.heatmap(s2)


accuracy=(cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,0]+cm[0,0]+cm[1,1])







