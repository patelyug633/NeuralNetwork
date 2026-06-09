# Neural Network from Scratch (Sequential → Batch-Vectorized GPU Implementation)

## Overview

This project is a custom neural network framework built entirely from scratch in Python to deeply understand how neural networks work at a fundamental level.

The project evolved in two major stages:

1. **Sequential Neural Network (Initial Prototype)**
2. **Batch-Vectorized GPU-Accelerated Neural Network (Optimized Version)**

This evolution highlights an important real-world machine learning insight:

> Correct implementations are not always efficient implementations.

The final system supports MNIST digit classification using GPU acceleration via CuPy and mini-batch training.

---

## Key Features

### Core Neural Network Implementation

* Fully custom feedforward neural network (no PyTorch / TensorFlow)
* Multi-layer perceptron support
* Manual forward and backward propagation
* ReLU activation (hidden layers)
* Softmax activation (output layer)
* Cross-entropy loss

### Training Capabilities

* Mini-batch gradient descent
* Momentum-based optimization
* He initialization for stable training
* GPU acceleration using CuPy

### Applied Functionality

* MNIST handwritten digit classification
* Real-time digit drawing interface for inference
* End-to-end training pipeline

---

# Project Evolution

## Phase 1: Sequential Neural Network (Initial Implementation)

The first version of this project implemented a fully sequential neural network using NumPy.

Each training sample was processed individually:

* Forward propagation per sample
* Backpropagation per sample
* Weight updates after full dataset iteration

### Example Use Case

The sequential model was successfully trained on simple logic tasks such as the AND gate, demonstrating correctness of backpropagation and learning behavior.

### Limitations Discovered

During scaling attempts, several major limitations were observed:

* Extremely slow training speed
* No vectorization or matrix optimization
* Poor hardware utilization
* Inefficient gradient computation
* Not scalable to datasets like MNIST

This phase was critical in validating correctness but exposed performance bottlenecks.

---

## Phase 2: Batch-Vectorized Neural Network (Optimization Phase)

To address performance limitations, the architecture was redesigned to support batch-based computation and GPU acceleration.

### Key Improvements

* Mini-batch training instead of single-sample updates
* Fully vectorized matrix operations using CuPy
* GPU acceleration via CUDA
* Momentum-based optimization (stable convergence)
* Efficient gradient aggregation across batches

### Result

The system was successfully scaled to train on the MNIST dataset with significantly improved performance.

---

## Key Technical Insight

A major takeaway from this project:

> A neural network can be mathematically correct but computationally unusable without proper vectorization.

This project demonstrates the transition from:

* **Correct but slow implementation**
  → to
* **Optimized, scalable deep learning system**

---

## Architecture

### BV_Layer (Batch Vectorized Layer)

Handles:

* Forward propagation (`matmul + activation`)
* Backpropagation (vectorized gradient computation)
* Momentum storage
* Batch-based delta computation

### BV_Network

Responsible for:

* Layer orchestration
* Forward pass pipeline
* Backward propagation chain rule implementation
* Weight updates
* Loss computation

---

## Training Pipeline (MNIST)

### Preprocessing

* Images flattened to 784-dimensional vectors
* Normalization (0–1 scaling)
* One-hot encoding of labels

### Training Loop

1. Shuffle dataset
2. Split into mini-batches
3. Forward pass (GPU-accelerated)
4. Compute cross-entropy loss
5. Backpropagation across layers
6. Update weights using momentum SGD

---

## Example Usage

```python
from Batchvectorization import BV_Network

model = BV_Network(
    input_size=784,
    output_size=10,
    batch_size=64,
    hidden_counts=[16, 16],
    beta=0.9
)

output = model.forward_pass(batch_X)
loss = model.get_cost(batch_Y)

model.backward_pass(batch_Y)
model.update_weights(learning_rate=0.01)
```

---

## Training Dynamics

During training, I recorded both:

- Cross-Entropy Loss
- Average Gradient Magnitude

The loss curve demonstrates successful optimization, while the gradient magnitude plot shows that learning signals continue propagating through the network during training.

![Training Curves](assets\training\mnist_neural_network_training.png)

---

## Results

* Successfully learns MNIST digit classification
* Loss decreases consistently during training
* Significant performance improvement over sequential version
* Demonstrates generalization to unseen test data

---

## Interactive Feature

The project includes a **real-time digit drawing application**:

* Users can draw digits on a canvas
* The model predicts the digit in real time
* Demonstrates practical inference capability beyond training

---

## Limitations

* No automatic differentiation engine
* Limited to fully connected networks (no CNN support yet)
* Requires GPU (CuPy) for optimal performance
* No model serialization (save/load not implemented yet)
* Manual gradient computation only

---

## What This Project Demonstrates

This project demonstrates understanding of:

* Neural network fundamentals from first principles
* Backpropagation and gradient computation
* Batch vectorization and performance optimization
* GPU-accelerated machine learning pipelines
* Real-world dataset training (MNIST)
* Engineering tradeoffs between correctness and efficiency

---

## Key Learning Outcome

This project evolved through a critical engineering realization:

> Early correct implementations are often not scalable implementations.

By transitioning from a sequential model to a batch-vectorized GPU system, the project demonstrates both:

* Theoretical correctness (learning capability)
* Practical scalability (performance optimization)

---

## Future Improvements

* Add convolutional neural network (CNN) support
* Implement model saving/loading (serialization)
* Build automatic differentiation engine (experimental)
* Improve drawing app preprocessing pipeline
* Add training visualization (loss/accuracy graphs)
* Extend to more complex datasets beyond MNIST

---

## Project Structure

```
NeuralNetwork/

│
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
│   ├── notes/
        ├── (Notes I took during building this project)
    ├── gui/
        ├── (Screenshots of my mnistdrawing app connected to the network predicting the digits)
│   
│
└── data/
    └── README.md
```

## Dataset

Download the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download) and place the files:

src/BatchVectorized_version/MnistData/


Required files:
- train-images-idx3-ubyte
- train-labels-idx1-ubyte
- t10k-images-idx3-ubyte
- t10k-labels-idx1-ubyte

---

## Author
Yug J. Patel. 
Computer Science student exploring deep learning from first principles.

GitHub: [Yug patel's Repositories](https://github.com/patelyug633?tab=repositories)
