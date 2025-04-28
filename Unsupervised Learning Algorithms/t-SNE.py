#t-SNE: t-Distributed Stochastic Neighbor Embedding (t-SNE)

#Import necessary libraries
from sklearn.manifold import TSNE
import numpy as np

#Sample data (eg: points in 3D space)
x = np.array([
    [1, 2, 1], [1.5, 1.8, 1.2], [5, 8, 6], [8, 8, 5], 
    [1, 0.6, 0.8], [9, 11, 8], [8, 2, 3], [10, 2, 2], [9, 3, 4]
])

#Initialize and fit the model
tsne = TSNE(n_components=2 , perplexity=5, random_state=42)
'''
1. n_components: reducing to 2D dimension
2. perplexity: is a parameter in tsne which controls the balance between local and global aspects of the data
'''
x_reduced = tsne.fit_transform(x)

print("Reduced Data: " , x_reduced)