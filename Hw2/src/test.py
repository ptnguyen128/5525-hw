import numpy as np

A = np.array([[1,2,3], [4,5,6],[7,8,9]])
y = np.array([1,0, 1])

K = y.reshape(-1,1)*A

print(y)