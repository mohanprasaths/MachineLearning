# -*- coding: utf-8 -*-
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

#import libraries
import numpy as np
import pandas as pd
from  io import StringIO
import matplotlib.pyplot as plt


dataset = pd.read_csv('data.csv')
dataset.dropna(inplace=True)
dataset = MultiColumnLabelEncoder(columns = ["Alcohol","Smoking","Gender", "Only child"]).fit_transform(dataset)

X = dataset.iloc[:,[0,19,36,71,73,106,123,140,141,142,144,147]].values
y = dataset.iloc[:,118].values
        




#Split training and test data               
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test,  = train_test_split(X,y,test_size= 0.2,random_state=0)

# Feature scalling
# To make the values of Independent variable in the same scale level
# using mean and standard deviation -
# Fit function find the mean and standard deviation , where transform applies them
# Training data - fit and transform - Test data - only transform
#
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

#Fit the regressor and use the mean and standard deviation in Prediction
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

myCalc = [5,5,	5,	3,	2,	3,	4,	20,	163,	48,	0,	0]


regressor.predict(myCalc)
