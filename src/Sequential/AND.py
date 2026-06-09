from Network import Network
import random
import numpy as np

np.random.seed(0)
random.seed(0)

def shuffle(X_train, Y_train):
    inds = [i for i in range(len(X_train))]
    random.shuffle(inds)
    newX = [X_train[i] for i in inds]
    newY = [Y_train[i] for i in inds]
    return newX , newY

X_train = [[1,1], [1,0], [0,0], [1,1], [0,1], [1,1], [1,0], [0,0], [1,1]]
Y_train = [1, 0, 0, 1, 0, 1, 0, 0, 1]

X = [[1,0], [1,1], [0,0], [0,1]]
Y = [0, 1, 0, 0]

epoch = 2000
print_every = 100

and_Net = Network(2, 1, 0.1)
print("Testing the Net before training")
print(f"Training accuracy: {and_Net.get_accuracy(X_train, Y_train):.2%}")
print(f"Validation accuracy: {and_Net.get_accuracy(X, Y):.2%}")
print(f"Dead hidden neurons: {and_Net.get_dead_neurons(X_train)}")

for i in range(len(X)):
    and_Net.forward_pass(X[i])
    predicted = and_Net.get_results()
    # print(f"For {X[i]}, net pridicted {predicted[0]}, actual value is {Y[i]}")
    print(f"For {X[i]}, net pridicted {round(predicted[0])}, actual value is {Y[i]}")

for i in range(epoch):
    cost = 0
    for j in range(len(X_train)):
        and_Net.forward_pass(X_train[j])
        cost += and_Net.get_cost(Y_train[j])
        and_Net.backward_pass(Y_train[j])
    
    cost /= 9
    and_Net.update_weights_with_momentum(len(X_train))

    if i % print_every == 0 or i == epoch - 1:
        train_accuracy = and_Net.get_accuracy(X_train, Y_train)
        val_accuracy = and_Net.get_accuracy(X, Y)
        dead_neurons = and_Net.get_dead_neurons(X_train)
        print(
            f"Batch #{i} finished, Cost = {cost:.6f}, "
            f"Train Acc = {train_accuracy:.2%}, "
            f"Val Acc = {val_accuracy:.2%}, "
            f"Dead hidden neurons = {dead_neurons}"
        )

    newX, newY = shuffle(X_train, Y_train)
    X_train = newX
    Y_train = newY

print("Now testing the Net")

for i in range(len(X)):
    and_Net.forward_pass(X[i])
    predicted = and_Net.get_results()
    # print(f"For {X[i]}, net pridicted {predicted[0]}, actual value is {Y[i]}")
    print(f"For {X[i]}, net pridicted {round(predicted[0])}, actual value is {Y[i]}")


# newX, newY = shuffle(X_train, Y_train)
