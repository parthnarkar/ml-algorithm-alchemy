#Principal Component Analysis (PCA)

#Import necessary libraries
from sklearn.decomposition import PCA
import numpy as np

#Sample data (eg: points in 3D space)
x = np.array([
    [1, 2, 1], [1.5, 1.8, 1.2], [5, 8, 6], [8, 8, 5], 
    [1, 0.6, 0.8], [9, 11, 8], [8, 2, 3], [10, 2, 2], [9, 3, 4]
])

#Initialize and fit the model
pca = PCA(n_components=2) #reducing to 2 dimensions
x_reduced = pca.fit_transform(x)

print("Reduced Data: ", x_reduced)
print("Explained Variance Ratio: " , pca.explained_variance_ratio_)
