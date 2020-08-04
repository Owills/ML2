import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn

names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','PRICE']

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                header=None, names=names , delim_whitespace = True, na_values='?')

df.head(5)

#print(df.shape)
y = df['PRICE'].values
x = df['RM'].values

plt.plot(x,y,'o')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Price')
plt.grid()

#line of best fit guess
#w1 = 9
#w0 = -30
#xplt = np.linspace(3,9,100)
#yplt = w1*xplt + w0
#plt.plot(x,y,'o')    # Plot the data points
#plt.plot(xplt,yplt,'-',linewidth=3)  # Plot the line
#yplt2=9.1*xplt -34.67
#yplt3=12.5*xplt-53
#plt.plot(xplt,yplt2,'-',linewidth=3,label='yplt2')
#plt.plot(xplt,yplt3,'-',linewidth=3,label='yplt3')
#plt.legend()

#yhat1 = 9*x-30
#yhat2 = 9.1*x-34.67
#yhat3 = 12.5*x-53

# Mean Squared Error

#MSE1 = np.mean((y-yhat1)**2)
#MSE2 = np.mean((y-yhat2)**2)
#MSE3 = np.mean((y-yhat3)**2)

#print(MSE1)
#print(MSE2)
#print(MSE3)
#mean absolute error
#MAE1 = np.mean(np.abs(y-yhat1))
#MAE2 = np.mean(np.abs(y-yhat2))
#MAE3 = np.mean(np.abs(y-yhat3))
#print(MAE1)
#print(MAE2)
#print(MAE3)

#get means, variance, covariance
print()
meanx = np.mean(x)
meany = np.mean(y)
#print(meanx)
#print(meany)
varx = np.var(x)
#print(varx)

covxy = np.mean((x-meanx)*(y-meany))
#print(covxy)

#least Square Soltuion
#solution: f(x) = meany + (covxy/varx)*(x-meanx)
#prediction: f(x) = w0 + w1*x
w1 = covxy/varx
w0 = meany-((covxy/varx)*meanx)
#print(w1)
#print(w0)
x_new = np.linspace(3,9,100)
y_new = w0+w1*x_new
plt.plot(x_new,y_new)
plt.title('Line of best fit')
print(np.mean((y-(w0+w1*x))**2))


#sklearn
#f(x) = w1*x + w0
regr = linear_model.LinearRegression()
regr.fit(x.reshape(-1,1),y.reshape(-1,1))

print('w1= ',regr.coef_[0], 'w0= ', regr.intercept_)
x_new = np.linspace(3,9,100)
y_new = regr.predict(x_new.reshape(-1,1))

plt.figure()
plt.plot(x_new,y_new, c='red', linewidth = 3)
plt.scatter(x,y)
plt.grid()
plt.title('Sklearn is your new best friend!');

yhat = regr.predict(x.reshape(-1,1))
print(sklearn.metrics.mean_squared_error(y, yhat))
plt.show()

#sample_weight

"""
Attribute Information:
    1.  CRIM      per capita crime rate by town
    2.  ZN        proportion of residential land zoned for lots over 
                  25,000 sq.ft.
    3.  INDUS     proportion of non-retail business acres per town
    4.  CHAS      Charles River dummy variable (= 1 if tract bounds 
                  river; 0 otherwise)
    5.  NOX       nitric oxides concentration (parts per 10 million)
    6.  RM        average number of rooms per dwelling
    7.  AGE       proportion of owner-occupied units built prior to 1940
    8.  DIS       weighted distances to five Boston employment centres
    9.  RAD       index of accessibility to radial highways
    10. TAX       full-value property-tax rate per $10,000
    11. PTRATIO   pupil-teacher ratio by town
    12. B         1000(Bk - 0.63)^2 where Bk is the proportion of blocks by town
    13. LSTAT     % lower status of the population
    14. MEDV      Median value of owner-occupied homes in $1000's
"""