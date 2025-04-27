#LINEAR REGRESSION

#Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split #this is used to split the datasets in training and testing sets
from sklearn.metrics import mean_squared_error
import numpy as np

#Sample data (eg: house size vs house price)
x = np.array([[1400], [1600], [1800], [2000], [2200], [2400], [2600]])
y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000])

#Split the data into training and testing sets
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2, random_state=42) 
'''
1. test_size here says that 20% of the data should be used for testing and rest of the data(80%) is used for training.
2. by using random_state=42 , can guarantee that the output of Run 1 will be equal to the output of Run 2, i.e. your split will be always the same.
'''

#Initialize and train the model
model = LinearRegression()
model.fit(x_train , y_train) #this finds the best fit for this training data

#Make predictions
y_pred = model.predict(x_test)

#Evaluate the model
mse = mean_squared_error(y_test , y_pred)
print("Mean Squared Error: ", mse)
print("Predicted Values: ", y_pred)