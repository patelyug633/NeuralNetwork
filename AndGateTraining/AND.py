from dataPros import MNISTreader as MNSTR
from Layers import Layer
from Network import Network
import numpy as np
import matplotlib.pyplot as plt  # Fixed import

reader = MNSTR()
reader.read("ANDTrain.csv")
Y,X = reader.getData()

reader.read("AndTest.csv")
tX = reader.getData(False)

reader.read("AndResult.csv")
tY = reader.getData(False)
print(tY)
# Create network (simpler architecture for AND gate)
h1 = Layer(2, 3, 0.1)  # 2 inputs, 3 neurons, LR=0.1
# h2 = Layer(3, 2, 0.1)
output = Layer(3, 1, 0.1, isOutput=True)  # 3 inputs from h1, 1 output
Net = Network([h1,output], "accumulated")  # Only 2 layers for simple problem

print("Performance before training:")
correct = 0
for i in range(len(tX)):
    prediction = Net.forward_pass(tX[i])  # Use tX[i], not tX[1]
    r = 1 if prediction[0] > 0.5 else 0  # Threshold at 0.5
    if tY[i] == r:
        correct += 1
    print(f"Input: {tX[i]}, Pred: {prediction[0]:.4f}, Rounded: {r}, True: {tY[i]}")

print(f"Accuracy: {(correct/len(tX))*100:.2f}%\n\n")

# Training loop
losses = []
epochs = 5000  # More epochs for convergence

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(len(X)):
        Net.forward_pass(X[i])
        loss = Net.get_loss(Y[i])  # Call as function
        epoch_loss += loss
        Net.backward_pass(Y[i])
        Net.update_weights()
    
    losses.append(epoch_loss / len(X))
    
    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {losses[-1]:.6f}")

# Plot training loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss for AND Gate")
plt.show()

print("\nPerformance after training:")
correct = 0
for i in range(len(tX)):
    prediction = Net.forward_pass(tX[i])
    r = 1 if prediction[0] > 0.5 else 0
    if tY[i] == r:
        correct += 1
    print(f"Input: {tX[i]}, Pred: {prediction[0]:.4f}, Rounded: {r}, True: {tY[i]}")

print(f"Accuracy: {(correct/len(tX))*100:.2f}%")

# Test on training data
print("\nTesting on training data:")
for i in range(len(X)):
    prediction = Net.forward_pass(X[i])
    r = 1 if prediction[0] > 0.5 else 0
    print(f"Input: {X[i]}, Pred: {prediction[0]:.4f}, Rounded: {r}, True: {Y[i]}")

print(f"Accuracy: {(correct/len(tX))*100:.2f}%")