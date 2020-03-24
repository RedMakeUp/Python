import numpy as np
import matplotlib.pyplot as plt

# Generate data
N = 100# Number of points per class
D = 2# Dimentionality
K = 3# Number of classes
X = np.zeros((N*K,D))# Data matrix(each row = single example)
n_example = X.shape[0]
y = np.zeros(N*K, dtype='uint8')# Class labels for each example
for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1.0, N)
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N) * 0.2
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

# Initialize the parameters
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))

# Some hyperparameters
reg = 1e-3# Regularization strength
step_size = 1e-0

# Gradient descent loop
for i in range(200):
    # Compute the class scores
    scores = np.dot(X, W) + b

    # Compute the loss
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1,keepdims=True)
    correct_logprobs = -np.log(probs[range(n_example), y])
    data_loss = np.sum(correct_logprobs) / n_example
    reg_loss = 0.5 * reg * np.sum(W*W)
    loss = data_loss + reg_loss
    if i % 10 == 0:
        print("iteration %d: loss %f" % (i, loss))


    # Compute the gradient
    dscores = probs
    dscores[range(n_example), y] -= 1
    dscores /= n_example
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    dW += reg * W

    # Update parameters
    W += -step_size * dW
    b += -step_size * db

# Evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print("Training accuracy: %.2f" % (np.mean(predicted_class == y)))

# Plot the resulting classifier
h = 0.02
x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='viridis')
plt.colorbar()
plt.show()