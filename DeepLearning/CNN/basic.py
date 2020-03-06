import numpy as np

# X is a 5x5x3 array
shape_x = (5, 5, 3)
X = np.array([
        [[0, 0, 1], [0, 2, 0], [1, 0, 2], [0, 2, 2], [1, 1, 0]],
        [[2, 1, 0], [1, 1, 0], [0, 1, 1], [0, 0, 0], [1, 2, 0]],
        [[2, 0, 0], [0, 1, 1], [2, 0, 1], [2, 2, 0], [0, 1, 0]],
        [[1, 2, 0], [1, 2, 0], [1, 0, 0], [1, 1, 0], [0, 2, 1]],
        [[0, 2, 0], [0, 2, 1], [1, 2, 2], [1, 1, 0], [0, 0, 2]]
    ])

# Add 1 zero-padding around X at exch depth slice
padding = 1
shape_y = (shape_x[0] + 2 * padding, shape_x[1] + 2 * padding, 3)
Y = np.zeros(shape = shape_y, dtype = np.int32)
for i in range(shape_y[0]):
    for j in range(shape_y[1]):
        for k in range(shape_y[2]):
            if i != 0 and i != shape_y[0] - 1 and j !=0 and j != shape_y[1] - 1:
                Y[i,j,k] = X[i-1, j-1, k]

# Filter weights(3x3x3), here holds a list of filters
shape_filter = (3, 3, 3)
W_0 = np.zeros(shape = (3, 3, 3), dtype = np.int32)
W_0[:, :, 0] = [
    [0, 0, -1],
    [0, -1, 1],
    [-1, -1, 1]
]
W_0[:, :, 1] = [
    [1, 1, -1],
    [-1, 1, -1],
    [1, 0, -1]
]
W_0[:, :, 2] = [
    [1, 0, 1],
    [0, 1, 0],
    [0, 0, 0]
]
W_1 = np.zeros(shape = (3, 3, 3), dtype = np.int32)
W_1[:, :, 0] = [
    [-1, 1, -1],
    [0, -1, 1],
    [0, -1, -1]
]
W_1[:, :, 1] = [
    [-1, 1, 0],
    [1, 1, 0],
    [1, -1, 1]
]
W_1[:, :, 2] = [
    [-1, 0, 0],
    [0, 1, 1],
    [0, 1, 0]
]
W = list()
W.append(W_0)
W.append(W_1)

# Bias for two filters 
bias = [1, 0]

# Convolve
stride = 2
output_size = int((shape_x[0] - shape_filter[0] + 2 * padding) / stride + 1)
shape_output = (output_size, output_size, 2)
output = np.zeros(shape = (output_size, output_size, 2), dtype = np.int32)
for i in range(output_size):
    for j in range(output_size):
        for k in range(2):
            output[i, j, k] = 0
            for d in range(shape_filter[2]):
                a = Y[i * stride: i * stride + 3, j * stride: j * stride + 3, d]
                b = W[k][:, :, d]
                output[i, j, k] += np.sum(a * b)
            output[i, j, k] += bias[k]

print(output[:, :, 0])
print(output[:, :, 1])