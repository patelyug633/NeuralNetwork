import numpy as np

class Layer:

    def __init__(self, prev_count, neuron_count, ind, isOutput):
        self.prev_count = prev_count
        self.neuron_count = neuron_count
        self.ind = ind
        self.isOutput = isOutput
        self.z = np.zeros(neuron_count)
        self.activations = np.zeros(neuron_count)
        self.delta = np.zeros(neuron_count)
        self.bias = np.zeros(neuron_count)
        self.delta_W = np.zeros((neuron_count, prev_count))
        self.delta_bias = np.zeros(neuron_count)
        self.previous_activations = np.zeros(prev_count)
        self.momentum_weights = np.zeros((neuron_count, prev_count))
        self.momentum_bias = np.zeros(neuron_count)

        # Rows are neurons in this layer, columns are neurons in the previous layer.
        if self.isOutput:
            self.w = np.random.randn(neuron_count, prev_count) * np.sqrt(2 / (prev_count + neuron_count))
        else:
            self.w = np.random.randn(neuron_count, prev_count) * np.sqrt(2 / prev_count)

    def get_activations(self, previous_activations):
        self.previous_activations = np.array(previous_activations)
        self.z = np.matmul(self.w, self.previous_activations) + self.bias
        if self.isOutput:
            self.activations = self.sigmoid(self.z)
        else:
            self.activations = self.ReLU(self.z)

        return self.activations
    
    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, x):
        return (x > 0).astype(int)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sigmoid_value = self.sigmoid(x)
        return sigmoid_value * (1 - sigmoid_value)
    
    def get_delta(self, next_layer_delta=None, next_layer_weights=None, actual_value=None):
        has_actual_value = actual_value is not None
        has_next_layer = next_layer_delta is not None and next_layer_weights is not None

        if self.isOutput:
            if not has_actual_value:
                raise ValueError("Output layers need actual_value to calculate delta.")
            if has_next_layer:
                raise ValueError("Output layers should not use next_layer_delta or next_layer_weights.")

            actual_value = np.array(actual_value)
            error = self.activations - actual_value
            activation_derivative = self.sigmoid_derivative(self.z)
        else:
            if has_actual_value:
                raise ValueError("Hidden layers should use next_layer_delta and next_layer_weights, not actual_value.")
            if not has_next_layer:
                raise ValueError("Hidden layers need both next_layer_delta and next_layer_weights.")

            error = np.matmul(next_layer_weights.T, next_layer_delta)
            activation_derivative = self.ReLU_derivative(self.z)

        self.delta = error * activation_derivative

        return self.delta

    def calc_gradients(self):
        delta_W = np.outer(self.delta, self.previous_activations)
        delta_bias = self.delta

        return delta_W, delta_bias
