#Import necessary Libraries
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
import numpy as np

#Sample Data (eg: hours studied and prior grades vs. pass/fail)
x = np.array([[1, 60],[2, 65],[3, 70],[4, 68],[5, 72],[6, 75],[7, 78],[8, 80],[9, 85],[10, 90]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

#Split the data into training and testing sets
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2, random_state=42) 

#Initialize and train the model 
model = SVC(kernel='linear') #'linear' kernel for linear classification
model.fit(x_train,y_train)

#Make predictions
y_pred = model.predict(x_test)

#Evaluate the model
accuracy = accuracy_score(y_test , y_pred)
conf_matrix = confusion_matrix(y_test , y_pred) #Confusion Matrix is contains false positive , false negative , true positive and true negative

#Print the Values
print("Accuracy: " , accuracy)
print("Confusion Matrix: " ,  conf_matrix)
