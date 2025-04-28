#DBSCAN: Density-Based Spatial Clustering of Applications w Noise

#Import necessary libraries
from sklearn.cluster import DBSCAN
import numpy as np 

#Sample data (eg: points in 2D space)
x = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8], 
    [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3]
])

#Initialize and fit the model
dbscan = DBSCAN(eps=3 , min_samples=2)
'''
1. eps: the maximum distance between two points to be considered.
2. min_samples: the minimum number of points required to form a dense region.
'''
dbscan.fit(x)

#Get the labels (-1 indicates noise)
labels = dbscan.labels_

print("Labels: " , labels)