#Import necessary libraries
from scipy.cluster.hierarchy import dendrogram , linkage
import matplotlib.pyplot as plt
import numpy as np

#Sample data (eg: points in 2D space)
x = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8], 
    [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3]
])

#Perform hierarcial/agglomerative clustering
z = linkage(x , method='ward') #'ward' minimizes variance within clusters

#Plot dendogram
plt.figure(figsize=(8,4))
dendrogram(z)
plt.title("Dendogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()