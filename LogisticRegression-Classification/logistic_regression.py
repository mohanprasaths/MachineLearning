# -*- coding: utf-8 -*-

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Getting data
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

#Split training and test data               
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test,  = train_test_split(X,y,test_size= 0.25,random_state=0)

# Feature scalling
# To make the values of Independent variable in the same scale level
# using mean and standard deviation -
# Fit function find the mean and standard deviation , where transform applies them
# Training data - fit and transform - Test data - only transform
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fit the logistic Regression by training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train , y_train)
                
#Predicting the test results
y_pred =  classifier.predict(X_test)