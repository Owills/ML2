import numpy as np


def sigmoid(x):
    x = np.array(x)
    return 1/(1+(np.exp(-x)))

print(sigmoid(1000))
print(sigmoid(-1000))
print(sigmoid(0))