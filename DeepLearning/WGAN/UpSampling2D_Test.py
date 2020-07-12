import numpy as np
import tensorflow as tf

# Define input data
X = np.asarray([
    [[1, 5], [2, 6]],
    [[4, 8], [3, 7]]
    ])
print(X)

# Sample * width * height * channel
X = X.reshape((1, 2, 2, 2))
print(X)

# Define model
model = tf.keras.Sequential()
model.add(tf.keras.layers.UpSampling2D(input_shape=(2, 2, 2), size=(2, 3)))
model.summary()

# Predict
Y = model.predict(X)
print(Y)
Y = Y.reshape((4, 6, 2))
Y_0 = Y[:, :, 0]
Y_1 = Y[:, :, 1]
print(Y_0)
print(Y_1)