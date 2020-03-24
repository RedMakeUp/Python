import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB

# X is sample points which have two coordinates
# y is an array of index of clusters
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
model = GaussianNB()
model.fit(X, y)

rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)

# Plot 
# ---------------------------------------------
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
ax.set_title('Naive Bayes Model', size = 14)

xlim = (-8, 8)
ylim = (-15, 5)
ax.set(xlim = (-6, 6), ylim = (-14, 4))

xg = np.linspace(xlim[0], xlim[1], 60)
yg = np.linspace(ylim[0], ylim[1], 40)
xx, yy = np.meshgrid(xg, yg)
Xgrid = np.vstack((xx.ravel(), yy.ravel())).T

for label, color in enumerate(['red', 'blue']):
    mask = (y == label)
    mu = X[mask].mean(axis = 0)
    std = X[mask].std(axis = 0)
    P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(axis = 1)
    Pm = np.ma.masked_array(P, P < 0.03)
    ax.pcolorfast(xg, yg, Pm.reshape(xx.shape), alpha = 0.5, cmap = color.title() + 's')
    ax.contour(xx, yy, P.reshape(xx.shape), levels = [0.01, 0.1, 0.5, 0.9], colors = color, alpha = 0.2)

ax.scatter(Xnew[:, 0], Xnew[:, 1], c = ynew, s = 20, cmap = 'RdBu', alpha = 0.1)

plt.show()
# ---------------------------------------------