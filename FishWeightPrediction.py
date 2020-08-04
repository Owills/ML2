import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

feature = pd.read_csv('https://raw.githubusercontent.com/huaijiangzhu/SummerML/master/day5/fish_market_feature.csv')
label = pd.read_csv('https://raw.githubusercontent.com/huaijiangzhu/SummerML/master/day5/fish_market_label.csv')
features = feature.columns.tolist()
labels = label.columns.tolist()

X = feature[features].values #print(X.shape) #124,5
Y = label[labels].values #print(Y.shape) #124,1
#print(Y)
ntrain = 96
nval = 28

inds = np.random.permutation(124)

train_choices = inds[:ntrain]
val_choices = inds[ntrain:ntrain+nval]

xtrain, ytrain = X[train_choices], Y[train_choices]
xval, yval = X[val_choices], Y[val_choices]

#print(xtrain.shape)
#print(xval.shape)
#exit()
def designMatrixM(x, M): #polynomial feature transformation
    x = x.reshape(-1,5)
    bias_col = np.ones((x.shape[0], 1))
    PhiX = bias_col
    for i in np.arange(1, M+1):
        PhiX= np.hstack([x, x**i])
    return PhiX

#def designMatrix(x):
 #   bias_term = np.ones((x.shape[0], 1))
  #  x = np.hstack([bias_term, x])
    #return x

Xtrain = designMatrixM(xtrain,1)
#XtrainM = designMatrixM(xtrain,25)
#print(xtrain.shape) #75,5
#print(Xtrain.shape) #75,6
#print(XtrainM.shape) #75, 126

# fitting the first model
reg = linear_model.LinearRegression(fit_intercept=True, normalize=False)
reg.fit(Xtrain, ytrain)

# training error
yhat = reg.predict(Xtrain)
MAE = np.mean(np.abs(ytrain-yhat))
print("Train MAE = %.4f" % MAE)

#adjusting the hyper parameters
BestM = 1
BestAlpha = 0
BestMAE = 99999
for i in range(1,25): #order
    Xtrain = designMatrixM(xtrain, i)
    Xval = designMatrixM(xval, i)
    for num in [float(j) / 100 for j in range(0, 100, 1)]: #alpha/lambda
        # fitting the model
        reg = linear_model.Ridge(alpha=num, fit_intercept=True, solver='lsqr', normalize=True)
        reg.fit(Xtrain, ytrain)

        # validation error
        yhat = reg.predict(Xval)
        MAE = np.mean(np.abs(yval-yhat))
        if(MAE < BestMAE):
            BestMAE = MAE
            BestM = i
            BestAlpha = num


print('Best M = ', BestM)
print('Best Alpha = ', BestAlpha)
print("Val MAE = %.4f" % BestMAE)

print()
# training error

#order 2, lamdda 0.01
#final model
Xtrain = designMatrixM(xtrain,BestM)
Xval = designMatrixM(xval, BestM)
X = designMatrixM(X, BestM)
reg = linear_model.Ridge(alpha=BestAlpha, fit_intercept=True, solver="lsqr", normalize=True)
reg.fit(Xtrain, ytrain)

yhat = reg.predict(Xtrain)
MAE = np.mean(np.abs(ytrain-yhat))
print("Train MAE = %.4f" % MAE)

yhat = reg.predict(Xval)
MAE = np.mean(np.abs(yval-yhat))
print("Val MAE = %.4f" % MAE)

#plt.show()

# test error
#Xtest = designMatrixM(xtest, BestM)
#yhat = reg.predict(Xtest)
#RMSE = np.sqrt(np.mean((ytest-yhat)**2))
#print("Test RMSE = %.4f" % RMSE)

X_test = pd.read_csv('https://raw.githubusercontent.com/huaijiangzhu/SummerML/master/day5/fish_market_test_feature.csv').values
y_test = pd.read_csv('https://raw.githubusercontent.com/huaijiangzhu/SummerML/master/day5/fish_market_test_label.csv').values
X_test = designMatrixM(X_test,BestM)
yhat = reg.predict(X_test)
MAE = np.mean(np.abs(y_test-yhat))
MSE =  np.mean((y_test-yhat)**2)
RMSE = np.sqrt( np.mean((y_test-yhat)**2) )
print()
print("Test MAE = %.4f" % MAE)
print("Test MSE = %.4f" % MSE)
print("Test RMSE = %.4f" % RMSE)
plt.plot(yhat,'o', label = 'yhat')
plt.plot(y_test,'o', label = 'ytrue')
plt.legend()
plt.show()