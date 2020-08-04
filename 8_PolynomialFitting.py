import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

url = 'https://raw.githubusercontent.com/huaijiangzhu/SummerML/master/day4/polyfit_data.csv'
df = pd.read_csv(url)
x = df['x'].values.reshape(-1,1)
y = df['y'].values.reshape(-1,1)
plt.plot(x,y,'o')
plt.figure()

##linear attempt, change degree to 1

#polynomial best fit using numpy polyfit
degree = 3 #more degrees better MSE, until you overfit
weights = np.polyfit(x.reshape(-1,), y.reshape(-1,), degree)
#with np.printoptions(precision=2):
    #print(weights)
#np.set_printoptions(suppress=True) #no scientific notation
#with np.printoptions(precision=2):
    #print(weights)

model = np.poly1d(weights)
yhat = model(x)

mse = np.mean((y-yhat)**2)
print('POLYNOMIAL DEGREEE = {}    '.format(degree), 'MSE = {}'.format(mse))

xline = np.linspace(np.min(x), np.max(x), 100)
yline = model(xline)
plt.plot(xline,yline,'b')
plt.plot(x,y,'o')
plt.show()
print()

#using  sklearn
#poly = PolynomialFeatures(3)
#poly.fit_transform(x)


#plt.show()