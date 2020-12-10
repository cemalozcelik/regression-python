import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns
import numpy as np
sns.set()
from sklearn.tree import DecisionTreeClassifier

ax = plt.gca()
X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)
ax.scatter(X[ :, 0 ], X[ :, 1 ], c=y, s=30, cmap='rainbow',
           clim=(y.min(), y.max()), zorder=3)
ax.axis('tight')
ax.axis('off')
xlim = ax.get_xlim()
ylim = ax.get_ylim()

model=DecisionTreeClassifier()
# fit the estimator
model.fit(X, y)
xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                     np.linspace(*ylim, num=200))
Z = model.predict(np.c_[ xx.ravel(), yy.ravel() ]).reshape(xx.shape)

# Create a color plot with the results
n_classes = len(np.unique(y))
contours = ax.contourf(xx, yy, Z, alpha=0.3,
                       levels=np.arange(n_classes + 1) - 0.5,
                       cmap='rainbow', clim=(y.min(), y.max()),
                       zorder=1)
ax.set(xlim=xlim, ylim=ylim)
plt.show()