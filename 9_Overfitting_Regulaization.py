import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
#matplotlib inline

nsamp = 25 # number of samples taken in all datasets
p = np.array([5,1,-2,-.5]) # true coefficients
#5x^3 + x^2 -2x - 0.5

var = 0.1 # noise variance

# we'll take a set of measurements uniformly
x = np.linspace(-1,1,nsamp)
y_true = np.polyval(p,x)
# noisy measurement, ym. use sqrt(var) as numpy normal standard deviation
y = y_true + np.random.normal(0, np.sqrt(var), nsamp)

plt.plot(x,y_true)
# we can force a scatter plot in plt.plot by making the third argument 'o'
plt.plot(x,y,'ob',markeredgecolor='black');
plt.grid();
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-1,1])
plt.ylim([-3,4])
plt.legend(['True Process, y_true','Noisy Measurement, y']);
# train test validation split
ntrain = 15
nval = 5
ntest = 5

inds = np.random.permutation(nsamp) #random samples

train_choices = inds[:ntrain]
val_choices = inds[ntrain:ntrain+nval]
test_choices = inds[ntrain+nval:]

xtrain, ytrain = x[train_choices], y[train_choices]
xval, yval     = x[val_choices], y[val_choices]
xtest, ytest   = x[test_choices], y[test_choices]

# forming the design matrix
# features x, model order M
def designMatrix(x, M):
    x = x.reshape(-1,1)
    bias_col = np.ones((x.shape[0], 1))
    PhiX = bias_col
    for i in np.arange(1, M+1):
        PhiX = np.hstack([PhiX, x**i])
    return PhiX


M = 25
Xtrain = designMatrix(xtrain, M)
#print(Xtrain.shape)

#No regularization
# fitting the model
reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit(Xtrain, ytrain)
w = reg.coef_

# training error
yhat = reg.predict(Xtrain)
RMSE = np.sqrt( np.mean((ytrain-yhat)**2) )
print("Train RMSE = %.4f" % RMSE)

# Validation error
Xval = designMatrix(xval, M)
yhat = reg.predict(Xval)
RMSE = np.sqrt( np.mean((yval-yhat)**2) )
print("Val RMSE = %.4f" % RMSE)

# plotting
x_line = np.linspace(-1,1,500).reshape(-1,1)
X_line = designMatrix(x_line, M)
y_line = reg.predict(X_line)

plt.figure()
plt.plot(x_line, y_line)
plt.plot(xtrain,ytrain,'o',markeredgecolor='black')
plt.plot(xtest,ytest,'o',markeredgecolor='black')
# plt.xlim([-1,1])
# plt.ylim([-3,3])
plt.legend(['Model','Train Points', 'Test Points'])
print("w = ")
with np.printoptions(precision=2, suppress=True):
    print(w.reshape(-1,1))


print()
# sklearn weight based regularization (squaring the weights in the cost function is called "ridge regression")
# fitting the model
reg = linear_model.Ridge(alpha=.05, fit_intercept=False, solver='cholesky') # aloha = lambda
reg.fit(Xtrain,ytrain)
w = reg.coef_
#w = np.linalg.inv(np.transpose(Xtrain) @ Xtrain + (lambda*np.eye(Xtrain.shape[0])) @ (np.transpose(Xtrain) @ ytrain)

# training error
yhat = reg.predict(Xtrain)
RMSE = np.sqrt( np.mean((ytrain-yhat)**2) )
print("Train RMSE = %.4f" % RMSE)

# validation error
Xval = designMatrix(xval, M)
yhat = reg.predict(Xval)
RMSE = np.sqrt( np.mean((yval-yhat)**2) )
print("Val RMSE = %.4f" % RMSE)

# plotting
x_line = np.linspace(-1,1,500).reshape(-1,1)
X_line = designMatrix(x_line, M)
y_line = reg.predict(X_line)

plt.figure()
plt.plot(x_line, y_line)
plt.plot(xtrain,ytrain,'o',markeredgecolor='black')
plt.plot(xtest,ytest,'o',markeredgecolor='black')
plt.plot(xval,yval,'o',markeredgecolor='black')
# plt.xlim([-1,1])
# plt.ylim([-3,3])
plt.legend(['Model','Train Points', 'Test Points', 'Val Points'])

print("w = ")
with np.printoptions(precision=2, suppress=True):
    print(w.reshape(-1,1))

#print()
# test error
Xtest = designMatrix(xtest, M)
yhat = reg.predict(Xtest)
RMSE = np.sqrt( np.mean((ytest-yhat)**2) )
print("Test RMSE = %.4f" % RMSE)

plt.show()
