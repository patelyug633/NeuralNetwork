import random
import cupy as cp

class BV_Layer:
    def __init__(self, input_size, output_size, batch_size, is_Output=False):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.is_Output = is_Output
        self.inputs = cp.zeros((batch_size, input_size))
        self.outputs = cp.zeros((batch_size, output_size))
        self.deltas = cp.zeros((batch_size, output_size))
        self.biases = cp.zeros((1, output_size))
        self.delta_weights = cp.zeros((input_size, output_size))
        self.delta_biases = cp.zeros((1, output_size))
        self.momentum_weights = cp.zeros((input_size, output_size))
        self.momentum_biases = cp.zeros((1, output_size))

        # Rows are neurons in this layer, columns are neurons in the previous layer.
        if self.is_Output:
            self.w = cp.random.randn(self.input_size, self.output_size) * cp.sqrt(2 / (self.input_size + self.output_size))
        else:
            self.w = cp.random.randn(self.input_size, self.output_size) * cp.sqrt(2 / self.input_size)
    
    def get_activations(self, inputs):
        self.inputs = inputs
        self.outputs = cp.matmul(inputs, self.w) + self.biases
        if self.is_Output:
            self.outputs = self.softmax(self.outputs)
        else:
            self.outputs = self.ReLU(self.outputs)
    
    def ReLU(self, x):
        return cp.maximum(0, x)
    
    def ReLU_derivative(self, x):
        return (x > 0).astype(int)
    
    def softmax(self, x):
        exp_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))  # Stability improvement
        return exp_x / cp.sum(exp_x, axis=1, keepdims=True)
    
    def get_deltas(self, next_layer_deltas=None, next_layer_weights=None, actual_values=None):
        if self.is_Output:
            if actual_values is None:
                raise ValueError("Output layers require actual values to calculate deltas.")
            self.deltas = self.outputs - actual_values
        else:
            if next_layer_deltas is None or next_layer_weights is None:
                raise ValueError("Hidden layers require next layer deltas and weights to calculate deltas.")
            self.deltas = cp.matmul(next_layer_deltas, next_layer_weights.T) * self.ReLU_derivative(self.outputs)
    
    def calc_delta_weights(self):
        self.delta_weights = cp.matmul(cp.transpose(self.inputs), self.deltas) / self.batch_size
        self.delta_biases = cp.sum(self.deltas, axis=0, keepdims=True) / self.batch_size
    

class BV_Network:
    def __init__(self, input_size, output_size, batch_size, hidden_counts=[], beta=0.9):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.hidden_counts = hidden_counts
        self.layers = []
        self.beta = beta
        
        if hidden_counts:
            self.layers.append(BV_Layer(input_size, hidden_counts[0], batch_size))
        
            for i in range(1, len(hidden_counts)):
                self.layers.append(BV_Layer(hidden_counts[i-1], hidden_counts[i], batch_size))
            
            
            self.layers.append(BV_Layer(hidden_counts[-1], output_size, batch_size, is_Output=True))
        else:
            self.layers.append(BV_Layer(input_size, output_size, batch_size, is_Output=True))
    
    def forward_pass(self, input_data):
        activations = input_data
        for layer in self.layers:
            layer.get_activations(activations)
            activations = layer.outputs
        return activations
    
    def backward_pass(self, actual_values):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer.is_Output:
                layer.get_deltas(actual_values=actual_values)
            else:
                next_layer = self.layers[i + 1]
                layer.get_deltas(next_layer_deltas=next_layer.deltas, next_layer_weights=next_layer.w)
            layer.calc_delta_weights()
    
    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.momentum_weights = self.beta * layer.momentum_weights + (1 - self.beta) * layer.delta_weights
            layer.momentum_biases = self.beta * layer.momentum_biases + (1 - self.beta) * layer.delta_biases
            
            layer.w -= learning_rate * layer.momentum_weights
            layer.biases -= learning_rate * layer.momentum_biases
    
    def get_cost(self, actual_values):
        output_layer = self.layers[-1]
        predicted = output_layer.outputs
        cost = -cp.sum(actual_values * cp.log(predicted + 1e-8)) / actual_values.shape[0]  # Cross-entropy loss
        return cost
    
    #Dangerous, Forgets all training 
    def change_batch_size(self, new_size):
        self.__init__(self.input_size, self.output_size, new_size, self.hidden_counts, self.beta)