#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,[1]].values
y = dataset.iloc[:,2].values


#Split training and test data  
"""             
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test,  = train_test_split(X,y,test_size= 0.2,random_state=0)"""

#Fit the regressor and use the mean and standard deviation in Prediction
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

#Fit the polynomial regression 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly , y)

#Visualisig Linear regression
#Visualise the data
plt.scatter(X , y , color = 'red')
plt.plot(X , regressor.predict(X) , color="blue")
plt.title('Salary vs experience on training data')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.show()

#Visualise the Polyomial Regression
plt.scatter(X , y , color = 'red')
plt.plot(X , lin_reg2.predict(poly_reg.fit_transform(X)) , color="blue")
plt.title('Salary vs experience on training data')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.show()

#Predicting new results
lin_reg2.predict(poly_reg.fit_transform(6.5))