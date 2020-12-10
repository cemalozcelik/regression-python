from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

X, y, coefficients = make_regression(n_samples=200, n_features=1, n_informative=1, n_targets=1, noise=20,coef=True, random_state=1)

plt.scatter(X,y,edgecolors="red",color="pink")

lr = LinearRegression()
lr.fit(X, y)
ylin_pred = lr.predict(X)
print("linear score=",lr.score(X,y))

rr = Ridge(alpha=1000)
rr.fit(X, y)
yrr_pred = rr.predict(X)
print("ridge regression train score for alpha=5,:",rr.score(X,y))

reg = Lasso(alpha=100)
reg.fit(X, y)
ylas_pred = reg.predict(X)
print("lasso regression train score for alpha=5:",reg.score(X,y))

el=ElasticNet(alpha=0.004)
el.fit(X,y)
yel_pred=el.predict(X)
print("ElasticNet regression train score for alpha=5:",el.score(X,y))

plt.plot(1,label="blue=Linear")
plt.plot(X, ylin_pred, c='b')
plt.plot(1,label="orange=Ridge")
plt.plot(X, yrr_pred, c='orange')
plt.plot(1,label="green=Lasso")
plt.plot(X, ylas_pred, c='g')
plt.plot(1,label="red=ElasticNet")
plt.plot(X, yel_pred, c='r')
plt.legend(title="Regularization\nalpha=1",loc='lower right')
plt.show()