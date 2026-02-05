# main.py
import thread from threads
import numpy as np
import matplotlib.pyplot as plt
from Layers import Layer
from Network import Network
from dataPros import MNISTreader as MNSTR
import random

h1 = Layer(784, 16, 0.01)
h2 = Layer(16, 16, 0.01)
output = Layer(16,10, 0.01, True)

Net = Network([h1,h2,output], "accumulated")


reader = MNSTR()
reader.read("train.csv")

y, x = reader.getData()
x = x/255
losses = []
epoch = 1000

for e in range(epoch):
    batch = 32
    for i in range(len(x)):
        Net.forward_pass(x[i])
        Net.backward_pass(y[i])
        if (i%32) == 0:
            losses.append(Net.get_loss(y[i]))
            print
            Net.update_weights(batch)
        
    if epoch % 5 == 0:
        print(f"Epoch {e}, Loss: {losses[-1]:.6f}")


tx, ty = x[:301], y[0:301]



for n in range(len(tx)):
    tx.append(x[n])
    ty.append(y[n])

print("Performance")
ans = 0
for n in range(len(tx)):
    r = Net.forward_pass(tx[n])
    if r == ty[n]:
        ans += 1

print("Accuracy:- ", (ans/300)*100)
