import numpy as np
import os

print(os.getcwd())

data = np.genfromtxt('01-10min.csv', delimiter=',')
print(data[:,0])


