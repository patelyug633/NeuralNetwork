import numpy as np
from Layers import Layer

class Network:

    def __init__(self, input_count, output_count, LnR=None, hidden_counts=None, beta=0.9):
        self.input_count = input_count
        self.output_count = output_count
        self.hidden_counts = hidden_counts if hidden_counts is not None else []
        self.layers = []
        self.learning_Rate = 0.01 if LnR is None else LnR 
        self.beta = beta
        

        self._build_layers()

    def update_arch(self, hidden_counts):
        self.hidden_counts = hidden_counts
        self._build_layers()

    def _build_layers(self):
        self.layers = []

        layer_counts = self.hidden_counts + [self.output_count]
        prev_count = self.input_count

        for ind, neuron_count in enumerate(layer_counts):
            isOutput = ind == len(layer_counts) - 1
            layer = Layer(prev_count, neuron_count, ind, isOutput)
            self.layers.append(layer)
            prev_count = neuron_count
    
    def forward_pass(self, input_data):
        activations = input_data
        for layer in self.layers:
            activations = layer.get_activations(activations)
        return activations
    
    def backward_pass(self, actual_value):
        for i  in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            if layer.isOutput:
                layer.get_delta(actual_value=actual_value)
            else:
                next_layer = self.layers[i + 1]
                layer.get_delta(next_layer_delta=next_layer.delta, next_layer_weights=next_layer.w)

            delta_W, delta_bias = layer.calc_gradients()
            layer.delta_W += delta_W
            layer.delta_bias += delta_bias
        
    def update_weights(self, batch_samp_amount):
        if batch_samp_amount <= 0:
            raise ValueError("batch_samp_amount must be greater than 0.")

        for layer in self.layers:
            layer.delta_W /= batch_samp_amount
            layer.delta_bias /= batch_samp_amount

            layer.w -= (self.learning_Rate * layer.delta_W)
            layer.bias -=  (self.learning_Rate * layer.delta_bias)
            layer.delta_W.fill(0)
            layer.delta_bias.fill(0)
    
    def update_weights_with_momentum(self, batch_samp_amount):
        if batch_samp_amount <= 0:
            raise ValueError("batch_samp_amount must be greater than 0.")

        for layer in self.layers:
            layer.delta_W /= batch_samp_amount
            layer.delta_bias /= batch_samp_amount

            layer.momentum_weights = self.beta * layer.momentum_weights + layer.delta_W
            layer.momentum_bias = self.beta * layer.momentum_bias + layer.delta_bias

            layer.w -= self.learning_Rate * layer.momentum_weights
            layer.bias -= self.learning_Rate * layer.momentum_bias

            layer.delta_W.fill(0)
            layer.delta_bias.fill(0)

    def update_weights_EMA_style(self, batch_samp_amount):
        if batch_samp_amount <= 0:
            raise ValueError("batch_samp_amount must be greater than 0.")

        for layer in self.layers:
            layer.delta_W /= batch_samp_amount
            layer.delta_bias /= batch_samp_amount

            layer.momentum_weights = self.beta * layer.momentum_weights + (1 - self.beta) * layer.delta_W
            layer.momentum_bias = self.beta * layer.momentum_bias + (1 - self.beta) * layer.delta_bias

            layer.w -= self.learning_Rate * layer.momentum_weights
            layer.bias -= self.learning_Rate * layer.momentum_bias

            layer.delta_W.fill(0)
            layer.delta_bias.fill(0)
    
    def get_cost(self, actual):
        actual = np.array(actual)
        return 0.5 * np.sum((self.layers[-1].activations - actual) ** 2)
    
    def get_results(self):
        return self.layers[-1].activations

    def _prediction_to_label(self, prediction):
        prediction = np.array(prediction)
        if self.output_count == 1:
            return int(prediction[0] >= 0.5)

        return int(np.argmax(prediction))

    def _actual_to_label(self, actual):
        actual = np.array(actual)
        if actual.size == 1:
            return int(actual)

        return int(np.argmax(actual))

    def get_accuracy(self, X_data, Y_data):
        correct = 0

        for x, y in zip(X_data, Y_data):
            prediction = self.forward_pass(x)
            if self._prediction_to_label(prediction) == self._actual_to_label(y):
                correct += 1

        return correct / len(X_data)

    def get_dead_neurons(self, X_data):
        dead_counts = []

        for layer in self.layers:
            if layer.isOutput:
                continue

            active_neurons = np.zeros(layer.neuron_count, dtype=bool)
            for x in X_data:
                self.forward_pass(x)
                active_neurons |= layer.activations > 0

            dead_counts.append(int(layer.neuron_count - np.count_nonzero(active_neurons)))

        return dead_counts

    
    
