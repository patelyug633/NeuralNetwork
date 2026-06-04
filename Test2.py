import numpy as np
import  cupy as cp
import random

arr = []

for i in range(32):
    arr1 = []
    for i in range(2):
        arr1.append(random.random())
    arr.append(arr1)

weights = [[2, 3, 1], [1,2,3]]
gpu_weights = cp.array(weights)

print(gpu_weights.shape)
print(cp.array(arr).shape)
# print(cp.array(arr) @ gpu_weights)

