import numpy as np
import pandas as pd
import tkinter as tk
class MNISTreader:
    def __init__(self):
        self.data = []

    def read(self, filepath):
        data = pd.read_csv(filepath)
        self.data = data.values
        return self.data
    
    def getData(self, training = True):
        if training:
            labels = self.data[:, :1]
            Pixels = self.data[:, 1:]
            lables = self.vectorized(labels)
            return labels, Pixels
        else:
            return self.data


    def vectorized(self, n):
        vec = np.zeros((len(n), 10))
        for i in range(len(n)):
            vec[i][n[i][0]] = 1
        
        return vec

