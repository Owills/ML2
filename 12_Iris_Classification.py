import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

#3 different types of irises’ (Setosa, Versicolour, and Virginica).
# There are 4 features in the dataset: sepal length, sepal width, petal length, petal width.


# Loading the dataset
iris = datasets.load_iris()
X = iris.data[:,:2] # using only first two features
y = iris.target

(num_samples, num_features) = X.shape
print("num_samples, num_features", X.shape)

plt.rcParams['figure.figsize'] = [6, 4]
plt.plot(X[y==0,0], X[y==0,1], 'o', markerfacecolor=(1,0,0,1), markeredgecolor='black')
plt.plot(X[y==1,0], X[y==1,1], 'o', markerfacecolor=(0,1,0,1), markeredgecolor='black')
plt.plot(X[y==2,0], X[y==2,1], 'o', markerfacecolor=(0,0,1,1), markeredgecolor='black');

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width');

print()
#sklearn
# normalizing the data
Xs = preprocessing.scale(X)

# splitting the data into test and train
# We're not using ANY hyperparameters --- NO NEED FOR VALIDATION SET
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.33)#, random_state=42)

# sklearn does everything in the background:
#     - onehot encoding
#     - softmax output
logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
logreg.fit(X_train, y_train)

yhat = logreg.predict(X_train)
acc = np.mean(yhat == y_train)
print("Train Accuracy = %f" % acc)

yhat = logreg.predict(X_test)
acc = np.mean(yhat == y_test)
print("Test Accuracy  = %f" % acc)

print()
#DECISION BOUNDARY
### CODE FROM SKLEARN IRIS DEMO ###

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = Xs[:, 0].min() - .5, Xs[:, 0].max() + .5
y_min, y_max = Xs[:, 1].min() - .5, Xs[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(2, figsize=(6, 4))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(Xs[:, 0], Xs[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())


print()
#UNDERSTANdING THE CLASSIFICATIONS
yhat_probs = logreg.predict_proba(X_test)

np.set_printoptions(precision=2, suppress=True)

print(yhat.shape, yhat_probs.shape)
print(np.hstack([yhat.reshape(-1,1), yhat_probs]))

print()
#ATTEMPT AT POLYNOMIAL FEATURE ENGINEERING
poly = PolynomialFeatures(9)
Xpoly = poly.fit_transform(X)

print(Xpoly.shape)

X_train, X_test, y_train, y_test = train_test_split(Xpoly, y, test_size=0.33, random_state=42)

logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', max_iter=5000)
logreg.fit(X_train, y_train)

print(logreg.score(X_train, y_train))

yhat = logreg.predict(X_test)
acc = np.mean(yhat == y_test)
print("Accuracy on test data = %f" % acc)

print()
#ONEHOT ENCODING
#def onehot_enc(y,num_classes):
    #(N,K) = (y.shape[0], num_classes)
    #I = np.eye(K)
   # y_onehot = np.zeros((N,K))
   # for i in range(N):
    #    y_onehot[i,:] = I[y[i],:]
    #return y_onehot

#y_onehot = onehot_enc(y, 3)
#print(y_onehot)

plt.show()