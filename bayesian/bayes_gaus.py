import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
plt.show()

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)

rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(200, 2)
ynew = model.predict(Xnew)

plt.scatter(X.T[ 0], X.T[ 1], c=y, s=50, cmap='rainbow')
lim = plt.axis()
plt.scatter(Xnew.T[ 0], Xnew.T[ 1], c=ynew, s=20, cmap='rainbow', alpha=0.4)
plt.axis(lim)
print(ynew)
plt.show()

yprob = model.predict_proba(Xnew)
print(yprob[-8:])
print(yprob[-8:].round(2))