# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 08:49:09 2019

@author: Rupesh S
"""

import pandas as pd

df=pd.read_excel("D:\\Data_Train.xlsx")
df1=pd.read_excel("D:\\Test_set.xlsx")
df.isnull().any()
df.isnull().sum()
df['Additional_Info'].value_counts()
df.dropna(how = 'any', axis = 0)
df['Airline'].unique()
df.dtypes
pd.to_datetime(df.Date_of_Journey).dt.day

#Cleaning Data
df['Duration']=df.Duration.str.replace(' ',':')
df['Duration']=df.Duration.str.replace('h',"")
df['Duration']=df.Duration.str.replace('m','')


df[['Hours','minutes']]=df['Duration'].str.split(':',expand=True)
df['minutes']=df.minutes.fillna(0)
df['Hours']=df.Hours.astype(int)
df['minutes']=df.minutes.astype(int)
#df['Hours']=df['Hours']*60+df['minutes']
df['Hours']=df['Hours'].apply(lambda df:df*60)
df['Total_Minutes']=df['Hours'] + df['minutes']

#Getting dummie values
df=pd.get_dummies(df,columns=['Airline','Source','Destination','Total_Stops','Additional_Info'])
df=df.drop(columns=['Hours','minutes','Duration','Route','Dep_Time','Arrival_Time'], axis=1)
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'])

df=df.drop(columns=['Date_of_Journey'])
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
vif_cal(df,'Price')

df.shape
df.info()

df.corr()
import matplotlib.pyplot as plt
plt.scatter(x=df['Price'],y=df['Total_Minutes'], color='blue')
plt.scatter(x=df['Price'], y=df['Airline_Air Asia'], color='red')
import seaborn as sns
df.corr()
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), annot=True)
plt.show()



