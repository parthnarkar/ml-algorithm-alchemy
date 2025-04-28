#GMM: Gaussian Mixture Models

#Import necessary libraries
from sklearn.mixture import GaussianMixture
import numpy as np

#Sample data (eg: points in 2D space)
x = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8], 
    [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3]
])

#Initialize and fit the model
gmm = GaussianMixture(n_components=2 , random_state=42)
'''
 n_components: 
'''
gmm.fit(x)

#Get the cluster labels and probabilities
labels = gmm.predict(x)
probs = gmm.predict_proba(x)

print("Cluster Labels: ", labels)
print("Cluster Probabilities: ", probs)