#Logistic Regression is used for binary classification problems (eg: yes/no , spam/not spam).

#Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
import numpy as np

#Sample data (eg: hours studied vs. pass/fail)
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]) 
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # Pass (1) / Fail (0)

#Split the data into training and testing sets
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2, random_state=42) 

#Initialize and train the model 
model = LogisticRegression()
model.fit(x_train,y_train)

#Make predictions
y_pred = model.predict(x_test)

#Evaluate the model
accuracy = accuracy_score(y_test , y_pred)
conf_matrix = confusion_matrix(y_test , y_pred) #Confusion Matrix is contains false positive , false negative , true positive and true negative

#Print the Values
print("Accuracy: " , accuracy)
print("Confusion Matrix: " ,  conf_matrix)