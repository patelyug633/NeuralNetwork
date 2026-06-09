# Neural Network From Scratch

### Sequential Backpropagation → Batch-Vectorized GPU Neural Network

## Project Highlights

* Built a neural network framework entirely from scratch using Python
* Implemented forward propagation and backpropagation manually
* Started with a sequential educational implementation using NumPy
* Redesigned the framework into a batch-vectorized GPU implementation using CuPy
* Implemented ReLU, Softmax, Cross-Entropy Loss, He Initialization, and Momentum SGD
* Trained on the MNIST handwritten digit dataset
* Achieved **95.31% test accuracy**
* Built an interactive digit drawing application for real-time inference

---

## Overview

This project was created to understand neural networks from first principles rather than relying on existing deep learning frameworks.

Instead of using PyTorch or TensorFlow, every major component was implemented manually, including:

* Feedforward computation
* Backpropagation
* Gradient calculation
* Weight updates
* Activation functions
* Loss functions
* Mini-batch training

The project evolved through two major implementations:

1. A fully sequential neural network written with NumPy
2. A batch-vectorized GPU-accelerated neural network written with CuPy

The transition between these two versions became one of the most valuable lessons of the project:

> A neural network can be mathematically correct while still being computationally impractical.

---

# Why I Built This

My original goal was simple:

> Understand exactly how neural networks learn.

I started by building a neural network from scratch using only NumPy. The first version successfully learned simple tasks such as the AND logic gate and proved that my implementation of backpropagation was correct.

However, when I attempted to scale the network to the MNIST handwritten digit dataset, training became extremely slow.

At that point I encountered a common engineering problem:

The implementation was correct, but it was not efficient.

Rather than abandoning the project, I investigated how modern machine learning systems achieve their performance and discovered the importance of:

* Matrix vectorization
* Mini-batch processing
* GPU acceleration
* Efficient gradient computation

This led me to redesign the framework from the ground up.

The final result was a batch-vectorized neural network capable of training on MNIST using GPU acceleration while preserving the same underlying mathematical principles as the original implementation.

---

# Features

## Neural Network Components

* Fully connected feedforward architecture
* Multiple hidden layers
* ReLU activation
* Softmax output layer
* Cross-Entropy loss
* He weight initialization
* Momentum optimization

## Training Features

* Mini-batch gradient descent
* GPU acceleration using CuPy
* Batch-vectorized forward propagation
* Batch-vectorized backpropagation
* Real-time training visualization

## Applications

* MNIST handwritten digit classification
* Interactive digit drawing GUI
* Real-time model inference

---

# Project Evolution

## Phase 1 — Sequential Implementation

The first implementation focused entirely on correctness.

Each sample was processed independently:

1. Forward pass
2. Cost calculation
3. Backpropagation
4. Gradient accumulation
5. Weight update

This version was invaluable for understanding:

* Chain rule applications
* Backpropagation mechanics
* Gradient flow
* Neural network architecture

### Example

The network successfully learned logical operations such as the AND gate.

### Limitations

While mathematically correct, several bottlenecks quickly appeared:

* Slow training speed
* No vectorization
* Poor CPU utilization
* Inefficient gradient computation
* Inability to scale effectively to larger datasets

---

## Phase 2 — Batch Vectorized GPU Implementation

To overcome these limitations, the framework was redesigned around matrix operations.

### Major Improvements

* Mini-batch training
* Vectorized forward propagation
* Vectorized backpropagation
* GPU acceleration through CuPy
* Momentum-based optimization

### Result

Training performance improved dramatically while preserving the same underlying learning algorithm.

This redesign transformed the project from a learning prototype into a practical machine learning system.

---

# Architecture

## BV_Layer

Responsible for:

* Matrix-based forward propagation
* ReLU and Softmax activations
* Delta computation
* Gradient computation
* Momentum storage

## BV_Network

Responsible for:

* Layer management
* Forward pass execution
* Backward propagation
* Cost calculation
* Weight updates

---

# MNIST Training Pipeline

## Data Preprocessing

* Flatten 28×28 images into 784-dimensional vectors
* Normalize pixel values to [0,1]
* Convert labels into one-hot encoded vectors

## Training Procedure

1. Shuffle dataset
2. Create mini-batches
3. Forward pass
4. Cross-entropy loss computation
5. Backpropagation
6. Momentum weight updates
7. Repeat for all epochs

---

# Results

## Dataset

MNIST Handwritten Digits

## Model Architecture

784 → 16 → 16 → 10

## Hyperparameters

| Parameter         | Value         |
| ----------------- | ------------- |
| Batch Size        | 64            |
| Learning Rate     | 0.01          |
| Activation        | ReLU          |
| Output Activation | Softmax       |
| Loss Function     | Cross Entropy |
| Optimizer         | Momentum SGD  |

## Performance

| Metric        | Value  |
| ------------- | ------ |
| Test Accuracy | 95.31% |
| Hardware      | GPU    |
| Backend       | CuPy   |

---

# Training Dynamics

During training I recorded both loss values and average gradient magnitudes.

The loss curve decreases consistently while gradient magnitudes remain active throughout training, indicating stable optimization and continued learning signal propagation.

![Training Curves](assets/training/mnist_training_curves.png)

---

# Interactive Digit Recognition App

The project includes a graphical application that allows users to draw digits directly onto a canvas.

Features:

* Draw digits using a mouse
* Real-time digit prediction
* Confidence scores for all classes
* Visualization of network input

Example:

![GUI Demo](assets/gui/digit_prediction_demo.png)

---

# Project Structure

```text
NeuralNetwork/

├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── BatchVectorized/
│   │   ├── BatchVectorization.py
│   │   ├── Mnist.py
│   │   ├── Mnist1.py
│   │   ├── MnistLoader.py
│   │   └── MnistDrawingApp.py
│   │
│   └── Sequential/
│       ├── Network.py
│       ├── Layers.py
│       └── AND.py
│
├── assets/
│   ├── training/
│   │   └── mnist_training_curves.png
│   │
│   ├── gui/
│   │   └── digit_prediction_demo.png
│   │
│   └── notes/
│       └── development_notes/
│
└── data/
    └── README.md
```

---

# What This Project Demonstrates

* Neural network fundamentals
* Manual backpropagation implementation
* Matrix calculus in practice
* Mini-batch optimization
* GPU computing with CuPy
* Performance engineering
* Software refactoring and redesign
* Training on a real-world dataset

---

# Limitations

* No convolutional neural networks (CNNs)
* No automatic differentiation engine
* No model serialization
* Limited to fully connected architectures
* Minimal hyperparameter search

---

# Future Improvements

* Add CNN support
* Implement model save/load functionality
* Add automatic differentiation
* Extend to Fashion-MNIST and CIFAR-10
* Add learning rate scheduling
* Implement Adam optimizer
* Improve GUI preprocessing pipeline

---

# Author

Yug J. Patel

Computer Science Student interested in Machine Learning, AI Systems, and Performance-Oriented Software Engineering.

GitHub:
https://github.com/patelyug633

