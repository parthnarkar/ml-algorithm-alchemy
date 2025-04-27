#Import necessary libraries
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

#Sample Data
x = np.array([[1400], [1600], [1800], [2000], [2200], [2400], [2600]])
y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000])

#Split the datasets
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

#Ridge Regression
ridge_model = Ridge(alpha=1.0) #alpha controls the regularization strength
ridge_model.fit(x_train , y_train)
ridge_predict = ridge_model.predict(x_test)
ridge_mse = mean_squared_error(ridge_predict , y_test)
print("Ridge Model MSE: " , ridge_mse)

#Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_train , y_train)
lasso_predict = lasso_model.predict(x_test)
lasso_mse = mean_squared_error(y_test , lasso_predict)
print("Lasso Model MSE: ", lasso_mse)