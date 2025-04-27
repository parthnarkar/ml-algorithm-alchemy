#Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
import numpy as np

#Sample data (eg: experience vs salary)
#Now the relationship between x and y is non-linear (i.e polynomial)
x = np.array([[1], [2], [3], [4], [5], [6], [7]])
y = np.array([30000, 35000, 50000, 60000, 65000, 70000, 85000])

#Split the data into training and testing sets
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2, random_state=42) 

#Transform features into polynomial features
poly = PolynomialFeatures(degree=2) #2nd degree polynomial
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

#Initialize and train the model
model = LinearRegression()
model.fit(x_train_poly , y_train)

#Make predictions
y_pred = model.predict(x_test_poly)

#Evaluate the model 
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)
print("Predicted Values: ", y_pred)

