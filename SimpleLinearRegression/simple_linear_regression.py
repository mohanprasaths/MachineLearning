# -*- coding: utf-8 -*-

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
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

#Visualise the data
plt.scatter(X_train , y_train , color = 'red')
plt.plot(X_train , regressor.predict(X_train) , color="blue")
plt.title('Salary vs experience on training data')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test , y_test , color = 'red')
plt.plot(X_train , regressor.predict(X_train) , color="blue")
plt.title('Salary vs experience on test data')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.show()