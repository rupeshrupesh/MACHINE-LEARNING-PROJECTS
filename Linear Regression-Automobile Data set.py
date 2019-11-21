# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:22:35 2019

@author: Rupesh S
"""

import pandas as pd
import numpy as np
import seaborn as sns

#Setting dimension for plot
sns.set(rc={'figure.figsize':(11.8,8.27)})

df=pd.read_csv("D:\\cars_sampled.csv")

cars=df.copy() #Creating copy file

cars.info() #Structure of the dataset

cars.describe()#summary of data
pd.options.display.float_format='{:,.3f}'.format
cars.describe()

pd.set_option('display.max_columns', 500) #to display maximum set of column
cars.describe()

#Dropping unwanted column

col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col, axis=1)

#Removing the duplicate record

cars.drop_duplicates(keep='first', inplace=True) #470 duplicate records

#Data cleaning process

cars.isnull().sum() #no.of.missing finding

#Variable year-of-registration
yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration']>2018)
sum(cars['yearOfRegistration']<1950)
sns.regplot(x='yearOfRegistration', y='price', scatter=True, fit_reg=False, data=cars)
#Working range -1950 to 2018


#Variable price

price_count=cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(y=cars['price'])
sum(cars['price']>150000)
sum(cars['price']<100)
#Working range -100 to 150000

#Variable PowerPS
power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=False, data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)
#Working range -10 to 500


#Working with data range
cars=cars[(cars.yearOfRegistration<=2018) 
&(cars.yearOfRegistration>=1950)
&(cars.price >=100)
&(cars.price<=150000)
&(cars.powerPS >=10)
&(cars.powerPS <=500)] #6700 records are dropped

#Further to simplify -variable reduction
#COMBINING YEARoFREGISTRATION AND MONTHOFREGISTRATION

cars['monthOfRegistration']/=12

#creating a new variable age by adding month and year
cars['age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['age']=round(cars['age'],2)
cars['age'].describe()

#Dropping year and month
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'], axis=1)


#Visualization parameter

#Age
sns.distplot(cars['age'])
sns.boxplot(y=cars['age'])

#price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

# powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

#Visualizing the parameter after narrowing working range
#Age vs Price
sns.regplot(x='age', y='price', scatter=True, fit_reg=True, data=cars)
#car priced higher are newer
#with increase in age, decrease price
#however some cars are higher with increase in age


#PowerPS vs price
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=True, data=cars)

#Variable seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'], columns='count', normalize=True)
sns.countplot(x='seller',data=cars)
#fewer car have commerical --> insignificant


#Variable offertype
cars['offerType'].value_counts()
sns.countplot(x='offerType',data=cars)
#All cars have offer-->insignificant

#Variable abtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'], columns='count', normalize=True)
sns.countplot(x='abtest', data=cars)
#Equally distributed
sns.boxplot(x='abtest', y='price',data=cars)
#for every price value there is almost 50-50 distribution
#Does not affect distribution -->insignificant

#Variable vehicle Type
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'], columns='count', normalize=True)
sns.countplot(x='vehicleType', data=cars)
sns.boxplot(x='vehicleType',y='price',data=cars)
#8-types-limousine, small cars and station wagen max frequency
#Vehicle type affect price

#variable gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'], columns='count', normalize=True)
sns.countplot(x='gearbox', data=cars)
sns.boxplot(x='gearbox',y='price',data=cars)
#gearbox affects the prices


#variable model
cars['model'].value_counts()
pd.crosstab(cars['model'], columns='count', normalize=True)
sns.countplot(x='model', data=cars)
sns.boxplot(x='model',y='price',data=cars)
#cars are distributed over many models
#considered in model

#Variable Kilometer
cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometerl'], columns='count', normalize=True)
sns.countplot(x='kilometer', data=cars)
sns.boxplot(x='model',y='price',data=cars)
cars['kilometer'].describe()
sns.distplot(cars['kilometer'],bins=8,kde=True)
sns.regplot(x='kilometer',y='price',scatter=True,fit_reg=True,data=cars)
#Considering in modeling

#Variable fuel type
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'], columns='count', normalize=True)
sns.countplot(x='fuelType', data=cars)
sns.boxplot(x='fuelType', y='price', data=cars)
#Fuel Types are affecting the price

#Variable Brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'], columns='count', normalize=True)
sns.countplot(x='brand', data=cars)
sns.boxplot(x='brand', y='price', data=cars)
#Cars are distributed over many brand
#considering the modelling

#variable notRepairedDamage
#Yes- car is damaged but not rectified
#No- car is damaged but has been redtified
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'], columns='count',normalize=True)
sns.countplot(x='notRepairedDamage', data=cars)
sns.boxplot(x='notRepairedDamage',y='price', data=cars)
#As expected , that cars that require the damage to be repaired
#Fall under low price ranges


#Removing the insignificants variable from data

col=['seller','offerType','abtest']
cars=cars.drop(columns=col, axis=1)

#Correlation
cars.corr()
correlation=cars.corr()
round(correlation,3)

#Omitting the missing values
cars_omit=cars.dropna(axis=0)

#Converting categorical variable into dummy variable
cars_omit=pd.get_dummies(cars_omit, drop_first=True)

#Important necessary Library

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Model Building with omitted data

x1=cars_omit.drop(['price'], axis=1, inplace=False)
y1=cars_omit['price']

#plotting the varible price
prices=pd.DataFrame({'1.Before':y1,'2.After':np.log(y1)})
prices.hist()

#Transforming prices as a logarithmic value
y1=np.log(y1)

#Splitting the data into train and test
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3, random_state=3)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#Baseline model for omitted data
#finding the mean value for test data
base_pred=np.mean(y_test)

#repeating the same value till length of test data
base_pred=np.repeat(base_pred, len(y_test))

#Finding the RMSE
base_roor_mean_squared_error=np.sqrt(mean_squared_error(y_test, base_pred))

#Linear Regression model with omitted data

#setting intercept
lgr=LinearRegression(fit_intercept=True)

#Model
model_ling1=lgr.fit(x_train, y_train)

#predicting model on test data
cars_prediction_line1=lgr.predict(x_test)

#computing MSE and RMSE
line_mse1=mean_squared_error(y_test,cars_prediction_line1)
line_mse1=np.sqrt(line_mse1)

#R squared value
r2_lin_model_test1=model_ling1.score(x_test,y_test)
r2_lin_model_train1=model_ling1.score(x_train,y_train)

#Regression diagnostic - residual analysis
residuals=y_test-cars_prediction_line1
sns.regplot(x=cars_prediction_line1, y=residuals, scatter=True, fit_reg=True, data=cars)
residuals.describe()

----------------------------------------END----------------------------------------------------------------------------