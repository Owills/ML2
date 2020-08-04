import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

iris = datasets.load_iris()
X = iris.data # using only first two features
y = iris.target

(num_samples, num_features) = X.shape
print("num_samples, num_features", X.shape)

# normalizing the data
Xs = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.33)

logreg = LogisticRegression(C=1e8, solver='lbfgs', multi_class='multinomial')
logreg.fit(X_train, y_train)

yhat = logreg.predict(X_train)
acc = np.mean(yhat == y_train)
print("Train Accuracy = %f" % acc)

yhat = logreg.predict(X_test)
acc = np.mean(yhat == y_test)
print("Test Accuracy  = %f" % acc)

print()
#ATTEMPT AT POLYNOMIAL FEATURE ENGINEERING
#poly = PolynomialFeatures(9)
#Xpoly = poly.fit_transform(X)

#print(Xpoly.shape)

#X_train, X_test, y_train, y_test = train_test_split(Xpoly, y, test_size=0.33, random_state=42)

#logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', max_iter=5000)
#logreg.fit(X_train, y_train)

#print("Train Accuracy = %f" % logreg.score(X_train, y_train))

#yhat = logreg.predict(X_test)
#acc = np.mean(yhat == y_test)
#print("Accuracy on test data = %f" % acc)