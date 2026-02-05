#Layers.py
import numpy as np
np.random.seed(42)

class Layererror(Exception):
    pass

class Layer:
    def __init__(self, conToPrev, n = 2, LR = 0.01, isOutput = False):
        self.a = []
        if n < 2:
            self.a = np.array([0]*2)
        else:
            self.a = np.array([0]*n)
        
        self.weights = np.random.rand(len(self.a), conToPrev)*np.sqrt(2/(conToPrev + n))
        self.bias = np.zeros(len(self.a))
        self.prevInput = []
        self.LR = LR
        self.delta = np.zeros_like(self.a)
        self.isOutPut = isOutput
        self.grad_weights = None
        self.grad_bias = None



    def set_weights(self, con):
        np.random.seed(2)
        self.weights = np.random.rand(len(self.a), con)


    def getactivation(self, inputs):
        inputs = np.array(inputs)
        z = np.dot(self.weights, inputs) + self.bias
        self.a = self.sigmoid(z)
        self.prevInput = inputs

    # def sigmoid(self, input):
    #     x = np.clip(input, -50, 50)
    #     return 1/(1+np.exp(-x))    

    def sigmoid(self, z):
        # Stable sigmoid implementation
        z = np.clip(z, -50, 50)  # More reasonable clipping
        return np.where(z >= 0, 
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))
    
    def calc_delta_OPL(self, y):
        if self.isOutPut:
            error = self.a - y
            self.delta = (error * self.a * (1 - self.a))
        else:
            print("This layer is not a output layer")
        
    def calc_delta_HDL(self, next_layer_delta, next_layer_weights):
        self.delta = (np.dot(next_layer_weights.T, next_layer_delta)*self.a*(1-self.a))
    
    def update_grad_weights(self):
        self.grad_weights = np.outer(self.delta, self.prevInput)
        self.grad_bias = self.delta
    
    def accumulate_grad_weights(self):
        if self.grad_weights is None and self.grad_bias is None:
            self.grad_weights = np.outer(self.delta, self.prevInput)
            self.grad_bias = self.delta
        else:
            self.grad_weights += np.outer(self.delta, self.prevInput)
            self.grad_bias += self.delta

    def change_weights(self):
        self.weights -= self.LR * self.grad_weights
        self.bias    -= self.LR * self.grad_bias





    
    
