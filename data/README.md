## Results

Model Architecture:

784 → 16 → 16 → 10

Dataset:

MNIST

Training:

- Batch Size: 64
- Activation: ReLU
- Output Activation: Softmax
- Loss: Cross Entropy
- Optimizer: Momentum SGD

Performance:

- Test Accuracy: 95.31%
- GPU Accelerated with CuPy