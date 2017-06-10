#Decision Tree regression in non continuous model

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,[1]].values
y = dataset.iloc[:,2].values

#Fitting the Random forest regression to the data set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X,y)

#Predict few values
y_pred = regressor.predict(6.6)

#Visualise the data
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X , y, color = 'red')
plt.plot(X_grid , regressor.predict(X_grid) , color="blue")
plt.title('Salary vs experience on training data <Random Forest regression>')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
