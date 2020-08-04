import matplotlib.pyplot as plt
import numpy as np
import math
import statistics


#print('sine')
#x = np.linspace(-6,6,100)
#y = np.sin(x)
#plt.plot(x, y)

#mean = np.mean(x)
#var = np.var(amplitude)
#print(mean)
#print(var)

#print('scatter plot points')
#x =[0, 2, 5, 4]
#y = [1, 3, 2, 1]
#plt.scatter(x,y)
#print (np.mean([1, 3, 2, 1]))
#print (np.var([1, 3, 2, 1]))

#print('y= mx =b')
#x = np.linspace(-5,5,10)
#y = 2*x+1
#plt.plot(x,y)
#print (np.mean(y))
#print (np.var(y))


#print('polynomial')
#x = np.linspace(-3,3,100)
#y = x**3+2
#plt.plot(x,y)
#print (np.mean(y))
#print (np.var(y))

#print('exponential')
#x = np.linspace(-3,3,100)
#y = np.exp(-2*x)
#plt.plot(x,y)
#print (np.mean(y))
#print (np.var(y))


#print('Gaussian')
#mean = 0; std = 0.5; variance = np.square(std)
#x = np.arange(-6,6,.01)
#y = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))
#plt.plot(x,y)

#print ('sigmoid')
x = np.linspace(-6, 6, 100)
y = 1 / (1 + np.exp(-x))
plt.plot(x, y)

#print(mean)
#print(variance)

#y_ticks = np.arange(0, 10, 0.5)
#plt.yticks(y_ticks)
plt.grid(True, which='both')
plt.axis([-6, 6, -10, 10])
plt.show()