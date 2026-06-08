#This file trains and record the cost and Gradient magnitude on the graphs while the model gets better and better

from BatchVectorization import BV_Network as Network
import random
import numpy as np
import cupy as cp
from os.path import join
from Mnistloader import MnistDataloader
import matplotlib.pyplot as plt
from collections import deque
import time

def shuffle(X_train, Y_train):
    inds = [i for i in range(len(X_train))]
    random.shuffle(inds)
    newX = [X_train[i] for i in inds]
    newY = [Y_train[i] for i in inds]
    return newX, newY

def rep_Y(Y_train):
    newY = []
    for y in Y_train:
        newY.append(np.array([1 if i == y else 0 for i in range(10)]))
    return newY

class TrainingVisualizer:
    def __init__(self):
        self.costs = []
        self.epochs = []
        self.start_time = time.time()
        
        # Setup the plot
        plt.ion()  # Interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 5))
        self.fig.suptitle('MNIST Neural Network Training Progress', fontsize=16, fontweight='bold')
        
        # Cost plot
        self.ax1.set_title('Training Cost Over Time', fontsize=12, fontweight='bold')
        self.ax1.set_xlabel('Epoch', fontsize=10)
        self.ax1.set_ylabel('Cross-Entropy Cost', fontsize=10)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_facecolor('#f8f9fa')
        
        # Gradient magnitude plot
        self.ax2.set_title('Average Gradient Magnitude', fontsize=12, fontweight='bold')
        self.ax2.set_xlabel('Epoch', fontsize=10)
        self.ax2.set_ylabel('Gradient Magnitude', fontsize=10)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_facecolor('#f8f9fa')
        
        self.gradient_magnitudes = []
        
        plt.tight_layout()
        
    def update(self, epoch, cost, network=None):
        self.costs.append(cost)
        self.epochs.append(epoch)
        
        # Calculate gradient magnitudes if network is provided
        if network:
            total_grad_mag = 0
            num_layers = 0
            for layer in network.layers:
                grad_mag = cp.mean(cp.abs(layer.delta_weights)).item()
                total_grad_mag += grad_mag
                num_layers += 1
            avg_grad_mag = total_grad_mag / num_layers
            self.gradient_magnitudes.append(avg_grad_mag)
        
        # Clear and redraw
        self.ax1.clear()
        self.ax2.clear()
        
        # Update cost plot
        self.ax1.plot(self.epochs, self.costs, 'b-', linewidth=2, label='Training Cost')
        self.ax1.set_title('Training Cost Over Time', fontsize=12, fontweight='bold')
        self.ax1.set_xlabel('Epoch', fontsize=10)
        self.ax1.set_ylabel('Cross-Entropy Cost', fontsize=10)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_facecolor('#f8f9fa')
        
        # Add annotations
        min_cost = min(self.costs)
        min_epoch = self.epochs[self.costs.index(min_cost)]
        self.ax1.plot(min_epoch, min_cost, 'ro', markersize=8, label=f'Best: {min_cost:.4f}')
        self.ax1.legend(loc='upper right')
        
        # Add text box with stats
        elapsed_time = time.time() - self.start_time
        stats_text = f'Epoch: {epoch}\nCost: {cost:.6f}\nTime: {elapsed_time:.1f}s'
        self.ax1.text(0.02, 0.98, stats_text, transform=self.ax1.transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Update gradient plot if we have data
        if self.gradient_magnitudes:
            self.ax2.plot(self.epochs, self.gradient_magnitudes, 'g-', linewidth=2, label='Gradient Magnitude')
            self.ax2.set_title('Average Gradient Magnitude', fontsize=12, fontweight='bold')
            self.ax2.set_xlabel('Epoch', fontsize=10)
            self.ax2.set_ylabel('Gradient Magnitude', fontsize=10)
            self.ax2.grid(True, alpha=0.3)
            self.ax2.set_facecolor('#f8f9fa')
            self.ax2.set_yscale('log')  # Log scale for gradients
            self.ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def save_final_plot(self, filename='mnist_training_results.png'):
        plt.ioff()
        self.fig.suptitle('MNIST Neural Network Training Results\nSuccessfully Classified Handwritten Digits', 
                          fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as '{filename}'")
        return filename

# Load data
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

# Convert to GPU arrays
x_train = cp.array(x_train)
x_test = cp.array(x_test)
y_train_targets = cp.array(y_train_targets)
y_test_targets = cp.array(y_test_targets)

# Training parameters
epoch = 50
batch_samp_amount = 64
print_every = 1

# Initialize network
mnist_Net = Network(784, 10, 10, hidden_counts=[16, 16], beta=0.9)

# Initialize visualizer
visualizer = TrainingVisualizer()

print("="*60)
print("MNIST NEURAL NETWORK TRAINING")
print("="*60)
print(f"Network Architecture: 784 → 16 → 16 → 10")
print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
print(f"Epochs: {epoch}")
print(f"Batch size: {batch_samp_amount}")
print("="*60)

print("\nTesting the Net before training:")
predict = mnist_Net.forward_pass(x_test[:10])
predict = cp.asnumpy(predict)
a = 0
for i in predict:
    i = np.argmax(i)
    print(f"  Predicted {i}, actual is {cp.argmax(y_test_targets[a])}")
    a += 1

mnist_Net.change_batch_size(batch_samp_amount)

# Training loop
for i in range(epoch):
    indices = cp.random.permutation(len(x_train))
    x_train_shuffled = x_train[indices]
    y_train_targets_shuffled = y_train_targets[indices]
    
    epoch_cost = 0
    batch_count = 0
    
    for j in range(0, len(x_train), batch_samp_amount):
        batch_X = x_train_shuffled[j:j+batch_samp_amount]
        batch_Y = y_train_targets_shuffled[j:j+batch_samp_amount]
        
        mnist_Net.forward_pass(batch_X)
        epoch_cost += mnist_Net.get_cost(batch_Y)
        mnist_Net.backward_pass(batch_Y)
        mnist_Net.update_weights(0.01)
        batch_count += 1
    
    avg_cost = epoch_cost / batch_count
    
    # Update visualization
    visualizer.update(i + 1, avg_cost.item(), mnist_Net)
    
    if i % print_every == 0 or i == epoch - 1:
        print(f"Epoch {i+1:3d}/{epoch} | Cost: {avg_cost:.6f} | Progress: {(i+1)/epoch*100:.1f}%")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

# Evaluate on test set
correct = 0
total = 0
for i in range(0, len(x_test), batch_samp_amount):
    batch_x = x_test[i:i+batch_samp_amount]
    batch_y = y_test_targets[i:i+batch_samp_amount]
    predict = mnist_Net.forward_pass(batch_x)
    for j in range(len(predict)):
        if cp.argmax(predict[j]) == cp.argmax(batch_y[j]):
            correct += 1
        total += 1

accuracy = (correct / total) * 100
print(f"\nFinal Test Accuracy: {accuracy:.2f}%")
print(f"Total correct: {correct}/{total}")

# Launch the drawing application
from MnistDrawingApp import launch_drawing_app
launch_drawing_app(mnist_Net)