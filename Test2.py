import numpy as np

weights = np.array([[1,5], [4,2], [3,4]])
inputs = np.array([[1, 2], [3,4],[3,6],[4,8]])


output = np.dot(inputs, weights.T)

print(output)