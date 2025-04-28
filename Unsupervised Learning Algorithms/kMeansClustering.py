#Import necessary libraries
from sklearn.cluster import KMeans
import numpy as np

#Sample data (eg: points in 2D space)
x = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8], 
    [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3]
])

#Initialize and fit the model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(x)

#Get the cluster center and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print("Cluster Centers: " , centroids)
print("Labels: " , labels)