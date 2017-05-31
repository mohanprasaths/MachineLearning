# -*- coding: utf-8 -*-

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values
        


# Encoding categorical data
# Encoding the Independent Variable
# Converting the label "string" values to number and make them categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Remove dummy variable trap
X = X[: , 1:]

#Split training and test data               
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test,  = train_test_split(X,y,test_size= 0.2,random_state=0)

#Fit the regressor and use the mean and standard deviation in Prediction
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)



