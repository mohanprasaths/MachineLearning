# -*- coding: utf-8 -*-

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
        
#Split training and test data               
from sklearn.cross_validation import train_test_split
X_test, X_train, y_test, y_train,  = train_test_split(X,y,train_size= 1/3,random_state=0)

#Fit the regressor and use the mean and standard deviation in Prediction
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)



