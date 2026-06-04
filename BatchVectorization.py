import random
import cupy as cp

class BV_Layer:
    def __init__(self, input_size, output_size, batch_size, isOutput=False):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.isOutput = isOutput
        self.input_activations = cp.zeros((batch_size, input_size))
        self.z = cp.zeros((batch_size, output_size))
        self.activations = cp.zeros((batch_size, output_size))
        self.delta = cp.zeros((batch_size, output_size))
        self.bias = cp.zeros(output_size)
        self.delta_W = cp.zeros((input_size, output_size))
        self.delta_bias = cp.zeros(output_size)

        # Rows are neurons in this layer, columns are neurons in the previous layer.
        if self.isOutput:
            self.w = cp.random.randn(self.input_size, self.output_size) * cp.sqrt(2 / (self.input_size + self.output_size))
        else:
            self.w = cp.random.randn(self.input_size, self.output_size) * cp.sqrt(2 / self.input_size)
    
    def get_activations(self, input_activations):
        self.input_activations = cp.array(input_activations)
        self.z = cp.matmul(self.input_activations, self.w) + self.bias
        if self.isOutput:
            self.activations = self.sigmoid(self.z)
        else:
            self.activations = self.ReLU(self.z)

        return self.activations
    
    def ReLU(self, x):
        return cp.maximum(0, x)
    
    def ReLU_derivative(self, x):
        return (x > 0).astype(int)
    
    def sigmoid(self, x):
        return 1 / (1 + cp.exp(-x))
    
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

            actual_value = cp.array(actual_value)
            error = self.activations - actual_value
            self.delta = error * self.sigmoid_derivative(self.z)
        else:
            if has_actual_value:
                raise ValueError("Hidden layers should not use actual_value to calculate delta.")
            if not has_next_layer:
                raise ValueError("Hidden layers need next_layer_delta and next_layer_weights to calculate delta.")

            self.delta = cp.matmul(next_layer_delta, cp.transpose(next_layer_weights)) * self.ReLU_derivative(self.z)
    
    def calc_gradients(self):
        delta_W = cp.matmul(cp.transpose(self.input_activations), self.delta)
        delta_bias = cp.sum(self.delta, axis=0)

        return delta_W, delta_bias


class BV_Network:
    def __init__(self, input_size, output_size, learning_Rate, hidden_counts=[], beta=0.9):
        self.learning_Rate = learning_Rate
        self.beta = beta
        self.layers = []
        layer_input_size = input_size

        for hidden_count in hidden_counts:
            self.layers.append(BV_Layer(layer_input_size, hidden_count, batch_size=0))
            layer_input_size = hidden_count
        
        self.layers.append(BV_Layer(layer_input_size, output_size, batch_size=0, isOutput=True))
    
    def forward_pass(self, batch_input):
        activations = batch_input
        for layer in self.layers:
            activations = layer.get_activations(activations)
        return activations
    
    def backward_pass(self, batch_actual):
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            if layer.isOutput:
                layer.get_delta(actual_value=batch_actual)
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
            layer.bias -= (self.learning_Rate * layer.delta_bias) 
            layer.delta_W.fill(0)
            layer.delta_bias.fill(0)
    
    def get_cost(self, batch_actual):
        output_layer = self.layers[-1]
        batch_actual = cp.array(batch_actual)
        batch_size = batch_actual.shape[0]
        cost = cp.sum((output_layer.activations - batch_actual) ** 2) / batch_size
        return cost
    
    def get_accuracy(self, X, Y):
        correct_predictions = 0
        for i in range(len(X)):
            predicted = self.forward_pass(X[i])
            predicted_label = int(cp.argmax(predicted))
            if predicted_label == Y[i]:
                correct_predictions += 1
        return correct_predictions / len(X)