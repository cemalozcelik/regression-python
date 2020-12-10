import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix

x=np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5])
y=np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1])

model=LogisticRegression().fit(x.reshape(-1,1),y)
b0=(model.intercept_[0])
b1=(model.coef_[0][0])
y1=model.predict_proba(x.reshape(-1,1))
y1=np.transpose(y1)

model = LinearRegression()
from sklearn.preprocessing import PolynomialFeatures

polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(x.reshape(-1,1))
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)


sigma=[]
for i in range(np.size(x)):
    sigma.append(1/(1+np.exp(-(x[i]*b1+b0))))

plt.plot(1,label="red=3th",c="r")
plt.plot(x,y_poly_pred,c="red")
plt.plot(x,y1[1],c="b")
plt.plot(1,label="blue=logistic",c="b")
plt.ylabel("probability of passing exam")
plt.xlabel("hours studying")
plt.legend()
plt.scatter(x,y)
plt.legend(title="Degree",loc='lower left')
plt.show()