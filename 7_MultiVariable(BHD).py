import pandas as pd
import numpy as np
from sklearn import linear_model
#import sklearn


names =['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE']

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                 header=None, delim_whitespace=True, names=names, na_values='?')

#Forming the Design Matrix
features = df.columns.tolist()
features.remove('PRICE')

#x = df[features[0]]
#xn = df[features[0]].values

#print(features)
#print(x[:3]) # pandas datatype
#print(xn[:3]) # numpy array

X = df[features].values
print(X.shape)

bias_term = np.ones((X.shape[0],1))
X = np.hstack([bias_term, X])
print(X.shape)

#LS Solution with sklearn
y = df['PRICE'].values
y=y.reshape(-1,1)
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X,y)
yhat = regr.predict(X)
#print(y_hat.shape)

#print(sklearn.metrics.mean_squared_error(y, yhat))
print('MSE: ', np.mean((y-yhat)**2))
print('COEF, W1 - WN: ', regr.coef_)
print('INTERCEPT, W0: ', regr.intercept_)

print()
#ground truth v model predictions
print("GROUND TRUTH v MODEL PREDICTIONS:")
Y = np.hstack([y, yhat])
with np.printoptions(precision=2):
    print(Y[:20,:])

print()
#Manuel LS Solution with pseudo-inverse
#X = df[features].values
##print(X.shape)
#bias_term = np.ones((X.shape[0],1))
#X = np.hstack([bias_term, X])
##print(X.shape)

# pseudo-inverse # X.dot(w) = X @ w
#w = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(y))

#ground truth v model predictions
#y_hat2 = X.dot(w)
#Y = np.hstack([y, y_hat2])
#with np.printoptions(precision=2):
#    print(Y[:10,:])

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