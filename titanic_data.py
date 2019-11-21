# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 05:35:39 2019

@author: Rupesh S
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("E:\\train.csv")
df.isnull().sum()
df.describe()
df.head()
df.info()
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='inferno')
sns.countplot(x='Survived', data=df)

df=df.dropna(subset=['Embarked'])

sns.countplot(x='Survived',hue='Sex', data=df)
sns.countplot(x='Survived', hue='SibSp',data=df)

sns.distplot(df['Age'].dropna(), bins=50, kde=False) #kernal density estimate
sns.boxplot(x='Pclass',y='Age', data=df)
sns.boxplot(x='Sex',y='Age', data=df)

df=df.drop(columns=['Cabin'])


for i in df.Pclass:
    if(i==1):
        df.Age.fillna(37, inplace=True)
    elif(i==2):
        df.Age.fillna(29, inplace=True)
    else:
        df.Age.fillna(24, inplace=True)
df=df.drop(columns=['Name','Ticket'])

df=pd.get_dummies(columns=['Sex','Embarked'], drop_first=True, data=df)


-----------------------------------------MULTICOLLINEARITY----------------------------------
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
vif_cal(df,'Survived')

x=df.drop(columns=['Survived'])
y=df.Survived.values

-------------------------------MODEL EVALUATION-----------------------------------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test, y_pred))
cm=confusion_matrix(y_test,y_pred)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,0]+cm[0,0]+cm[1,1])


----------------------------------TEST--------------------------------------------------------
df2=pd.read_csv("E:\\test.csv")

df2.isnull().sum()
df2=df2.drop(columns=['Cabin'])
df2=df2.drop(columns=['Name','Ticket'])
df2=pd.get_dummies(columns=['Sex','Embarked'], drop_first=True, data=df2)

sns.boxplot(x='Pclass',y='Age', data=df2)

for i in df2.Pclass:
    if(i==1):
        df2.Age.fillna(41, inplace=True)
    elif(i==2):
        df2.Age.fillna(28, inplace=True)
    else:
        df2.Age.fillna(24, inplace=True)
        
df2['Fare']=df2.Fare.fillna(df2.Fare.mean())        

y_pred1=classifier.predict(df2)


d=pd.DataFrame({'PassengerId':df2['PassengerId'],'Survived':y_pred1[:]})

