import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x=np.arange(-25,25).reshape(-1,1)
y=[]
for i in range(x.size):
    if i%2==0:
        y.append(x[i]**3/10-i*60)
    else:
        y.append(x[i] ** 4 / 200 -i*40)
y=np.array([1.132,2.069,2.465,2.578,2.636,2.671,2.697,2.718,2.731,2.742]).reshape(-1,1)
x=np.array([0.0001,0.0312,0.71,1.65,2.60,3.67,4.70,5.75,6.78,7.78]).reshape(-1,1)



from sklearn.preprocessing import PolynomialFeatures

linear_features= PolynomialFeatures(degree=1)
polynomial_features= PolynomialFeatures(degree=2)
polynomial_features2= PolynomialFeatures(degree=3)

x_poly = linear_features.fit_transform(x)
x_poly2 = polynomial_features.fit_transform(x)
x_poly3 = polynomial_features2.fit_transform(x)

model = LinearRegression()

model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

model.fit(x_poly2, y)
y_poly_pred2 = model.predict(x_poly2)

model.fit(x_poly3, y)
y_poly_pred3 = model.predict(x_poly3)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
rmse1 = np.sqrt(mean_squared_error(y,y_poly_pred2))
r21 = r2_score(y,y_poly_pred2)
rmse2 = np.sqrt(mean_squared_error(y,y_poly_pred3))
r22 = r2_score(y,y_poly_pred3)
print("1th rmse=",rmse)
print("1th degree r2=",r2)
print("2th rmse=",rmse1)
print("2th degree r2=",r21)
print("3th rmse=",rmse2)
print("3th degree r2=",r22)

plt.scatter(x, y, s=10)
plt.plot(x, y_poly_pred, color='blue')
plt.plot(1,label="blue=1th")
plt.plot(x, y_poly_pred2, color='orange')
plt.plot(1,label="orange=2th")
plt.plot(x, y_poly_pred3, color='green')
plt.plot(1,label="green=3th")
plt.legend(title="Degree",loc='lower left')
plt.show()