from BatchVectorization import BV_Network as Network
import random
import numpy as np
import cupy as cp
from os.path import join
from Mnistloader import MnistDataloader

def shuffle(X_train, Y_train):
    inds = [i for i in range(len(X_train))]
    random.shuffle(inds)
    newX = [X_train[i] for i in inds]
    newY = [Y_train[i] for i in inds]
    return newX , newY
def rep_Y(Y_train):
    newY = []
    for y in Y_train:
        newY.append(np.array([1 if i == y else 0 for i in range(10)]))
    return newY
input_path = 'C:\\Users\\patel\\Desktop\\Projects\\NeuralNetwork\\MnistData'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')


mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, 
                                   test_images_filepath, test_labels_filepath)

(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

x_train = [np.asarray(x, dtype=float).reshape(-1) / 255.0 for x in x_train]
x_test = [np.asarray(x, dtype=float).reshape(-1) / 255.0 for x in x_test]
y_train_targets = rep_Y(y_train)
y_test_targets = rep_Y(y_test)

x_train = cp.array(x_train)
x_test = cp.array(x_test)
y_train_targets = cp.array(y_train_targets)
y_test_targets = cp.array(y_test_targets)

epoch = 500
batch_samp_amount = 64
print_every = 1
mnist_Net = Network(784, 10, 10, hidden_counts=[16, 16], beta=0.9)
print("Testing the Net before training")
predict = mnist_Net.forward_pass(x_test[:10])
predict = cp.asnumpy(predict)
a = 0
for i in predict:
    i = np.argmax(i)
    print(f"Predicted {i}, actual is {cp.argmax(y_test_targets[a])}")
    a += 1

mnist_Net.change_batch_size(batch_samp_amount)
for i in range(epoch):
    indices = cp.random.permutation(len(x_train))
    x_train_shuffled = x_train[indices]
    y_train_targets_shuffled = y_train_targets[indices]  # Use the one-hot targets!

    cost = 0
    for j in range(0, len(x_train), batch_samp_amount):
        batch_X = x_train_shuffled[j:j+batch_samp_amount]
        batch_Y = y_train_targets_shuffled[j:j+batch_samp_amount]
        
        mnist_Net.forward_pass(batch_X)
        cost += mnist_Net.get_cost(batch_Y)
        mnist_Net.backward_pass(batch_Y)
        
        current_batch_size = len(batch_X)
        mnist_Net.update_weights(0.01)

    cost = cost / len(x_train)

    if i % print_every == 0 or i == epoch - 1:
        # train_accuracy = mnist_Net.get_accuracy(x_train, y_train_targets)
        # val_accuracy = mnist_Net.get_accuracy(x_test, y_test_targets)
        # dead_neurons = mnist_Net.get_dead_neurons(x_train)
        print(
            f"Batch #{i} finished, Cost = {cost:.6f}, "
            # f"Train Acc = {train_accuracy:.2%}, "
            # f"Val Acc = {val_accuracy:.2%}, "
            # f"Dead hidden neurons = {dead_neurons}"
        )

correct  = 0
total =  0
for i in range(0, len(x_test), batch_samp_amount):
    batch_x = x_test[:batch_samp_amount]
    batch_y = y_test_targets[:batch_samp_amount]
    predict = mnist_Net.forward_pass(batch_x)
    a = 0
    for j in predict:
        j = cp.argmax(j)
        if j == cp.argmax(y_test_targets[a]):
            correct += 1
        total += 1
        a += 1

print(f"Accuracy: {(correct/total)*100}%")

# Launch the drawing application
from MnistDrawingApp import launch_drawing_app  # Save the above code in drawing_app.py
launch_drawing_app(mnist_Net)