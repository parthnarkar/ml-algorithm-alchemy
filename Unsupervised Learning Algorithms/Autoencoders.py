#Autoencoders are neural networks used for unsupervised learning, specifically for dimensionality reduction and feature extraction.

#import necessary libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input , Dense
import numpy as np

#Sample data (eg: points in 5-dimensional space)
x = np.array([
    [1.0, 2.5, 3.1, 4.0, 5.2],
    [2.2, 3.3, 1.5, 4.8, 2.1],
    [0.4, 1.2, 3.6, 2.5, 4.9],
    [3.1, 0.9, 2.3, 5.6, 3.4],
    [1.9, 2.4, 0.7, 4.3, 5.1]
])

#Define the autoencoder model
input_dim = x.shape[1]
encoding_dim = 2 #Compressing to 2 dimensions

#Encoder 
input_layer = Input (shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)

#Decoder
decoded = Dense(input_dim , activation='sigmoid')(encoded)

#Autoencoder model
autoencoder = Model(input_layer , decoded)

#Compile the model 
autoencoder.compile(optimizer='adam' , loss='mse')

#Train the model
autoencoder.fit(x , x , epochs=100 , batch_size=2 , verbose=0)

#Get the encoded (compressed) representation
encoder = Model(input_layer , encoded)
x_compressed = encoder.predict(x)

print("Compressed Representation: ", x_compressed)