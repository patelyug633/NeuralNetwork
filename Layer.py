import numpy as np 


class Layer:

    def __init__(self, contoPrev, n, LR, isOutput = False):
        self.activation = np.zeros(n) if n > 0 else np.zeros(1)
        n = len(activation)
        self.weight = np.random.randn(contoPrev, n) * np.sqrt(1/contoPrev + n)
        self.bias = np.zeros_like(self.activation)