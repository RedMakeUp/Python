import numpy as np

# Softmax classifer gets its name from the softmax function
# For numeric stability, softmax fomular uses the expression from http://cs231n.github.io/linear-classify/#loss 
def Softmax(x):
    log_c = -np.max(x)
    a = np.exp(x + log_c)
    return a / np.sum(a)

# Softmax classifer uses cross-entropy loss
def Cross_entropy_loss(true_dist, predicted_dist):
    result = 0.0
    for i in range(len(true_dist)):
        result += true_dist[i] * np.log(predicted_dist[i])
    return -result

# -------------------------------------------------------------------------- #

scores = np.array([3.2, 5.1, -1.7])

print(Cross_entropy_loss([1,0,0], Softmax(scores)))
