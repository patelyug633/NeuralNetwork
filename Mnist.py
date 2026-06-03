from Network import Network
import random
import numpy as np
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

epoch = 2000
batch_samp_amount = 64
print_every = 100
mnist_Net = Network(784, 10, 0.1, hidden_counts=[16, 16], beta=0.9)
print("Testing the Net before training")
for i in range(10):
    predicted = mnist_Net.forward_pass(x_test[i])
    predicted_label = int(np.argmax(predicted))
    print(f"For test image {i}, net pridicted {predicted_label}, actual value is {y_test[i]}")


for i in range(epoch):
    cost = 0
    for j in range(0, len(x_train), batch_samp_amount):
        batch_X = x_train[j:j+batch_samp_amount]
        batch_Y = y_train_targets[j:j+batch_samp_amount]
        
        for k in range(len(batch_X)):
            mnist_Net.forward_pass(batch_X[k])
            cost += mnist_Net.get_cost(batch_Y[k])
            mnist_Net.backward_pass(batch_Y[k])
        
        current_batch_size = len(batch_X)
        mnist_Net.update_weights(current_batch_size)

    cost = cost / len(x_train)
    
    if i % print_every == 0 or i == epoch - 1:
        train_accuracy = mnist_Net.get_accuracy(x_train, y_train_targets)
        val_accuracy = mnist_Net.get_accuracy(x_test, y_test_targets)
        dead_neurons = mnist_Net.get_dead_neurons(x_train)
        print(
            f"Batch #{i} finished, Cost = {cost:.6f}, "
            f"Train Acc = {train_accuracy:.2%}, "
            f"Val Acc = {val_accuracy:.2%}, "
            f"Dead hidden neurons = {dead_neurons}"
        )

    newX, newY = shuffle(x_train, y_train_targets)
    x_train = newX
    y_train_targets = newY
