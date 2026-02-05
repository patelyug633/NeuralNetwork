#Network.py

import numpy as np

class Network:
    def __init__(self, Layers, state="normal"):
        self.Layers = Layers
        self.y_pred = np.zeros_like(self.Layers[-1])
        self.grad_state = state
        

    def forward_pass(self, x):
        ind = x
        n = len(self.Layers)

        for i in range(n):
            self.Layers[i].getactivation(ind)
            ind = self.Layers[i].a
        
        self.y_pred = ind
        return ind
    
    def backward_pass(self, y_true):
        self.Layers[-1].calc_delta_OPL(y_true)
        if self.grad_state == "accumulated":
            self.Layers[-1].accumulate_grad_weights()
        elif self.grad_state == "normal":
            self.Layers[-1].update_grad_weights()
        delta = self.Layers[-1].delta.copy()
        weights = self.Layers[-1].weights.copy()
        for i in range(len(self.Layers)-2, -1, -1):
            self.Layers[i].calc_delta_HDL(delta,weights)
            delta = self.Layers[i].delta.copy()
            weights = self.Layers[i].weights.copy()
            if self.grad_state == "accumulated":
                self.Layers[i].accumulate_grad_weights()
            elif self.grad_state == "normal":
                self.Layers[i].update_grad_weights()

    
    def update_weights(self, batchnum = 1):
        for Layer in self.Layers:
            Layer.grad_weights /= batchnum
            Layer.grad_bias /= batchnum
            Layer.change_weights()
            Layer.grad_weights = None
            Layer.grad_bias = None
    
    def get_loss(self, y_true):
        return 0.5 * np.mean((y_true - self.y_pred)**2)