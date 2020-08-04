import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model, preprocessing

names = ['id','thick','size','shape','marg','cell_size','bare',
         'chrom','normal','mit','class']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/' +
                 'breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                names=names,na_values='?',header=None)
df = df.dropna()
df.head(6)

yraw = np.array(df['class'])
BEN_VAL = 2   # value in the 'class' label for benign samples
MAL_VAL = 4   # value in the 'class' label for malignant samples
y = (yraw == MAL_VAL).astype(int)
Iben = (y==0)
Imal = (y==1)

# Get two predictors
xnames =['size','marg']
X = np.array(df[xnames])

# Create the scatter plot
plt.plot(X[Imal,0],X[Imal,1],'r.')
plt.plot(X[Iben,0],X[Iben,1],'g.')
plt.xlabel(xnames[0], fontsize=16)
plt.ylabel(xnames[1], fontsize=16)
plt.ylim(0,14)
plt.legend(['malign','benign'],loc='upper right')
plt.figure()

#The above plot is not informative, since many of the points
# are on top of one another. Thus, we cannot see the relative
# frequency of points.
#One way to improve the plot is to draw circles on each point
# whose size is proportional to the count of samples at that point.
# We will re-use this code, so we define a function.

def plot_count(X, y):
    # Compute the bin edges for the 2d histogram
    x0val = np.array(list(set(X[:, 0]))).astype(float)
    x1val = np.array(list(set(X[:, 1]))).astype(float)
    x0, x1 = np.meshgrid(x0val, x1val)
    x0e = np.hstack((x0val, np.max(x0val) + 1))
    x1e = np.hstack((x1val, np.max(x1val) + 1))

    # Make a plot for each class
    yval = list(set(y))
    color = ['g', 'r']
    for i in range(len(yval)):
        I = np.where(y == yval[i])[0]
        count, x0e, x1e = np.histogram2d(X[I, 0], X[I, 1], [x0e, x1e])
        x0, x1 = np.meshgrid(x0val, x1val)
        plt.scatter(x0.ravel(), x1.ravel(), s=2 * count.ravel(), alpha=0.5,
                    c=color[i], edgecolors='none')
    plt.ylim([0, 14])
    plt.legend(['benign', 'malign'], loc='upper right')
    plt.xlabel(xnames[0], fontsize=16)
    plt.ylabel(xnames[1], fontsize=16)
    return plt


plot_count(X, y)

plt.show()