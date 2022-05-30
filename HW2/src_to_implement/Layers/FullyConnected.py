import numpy as np
from Optimization import Optimizers
from Layers import Base

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(size = (input_size+1, output_size))
        self._optimizer = None
        self.temp = []

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self, input_tensor):
        self.lastIn = input_tensor
        last_col = np.ones((input_tensor.shape[0],1))
        self.lastIn = np.concatenate((self.lastIn, last_col), axis = 1)
        return np.dot(self.lastIn, self.weights)
        
    def backward(self, error_tensor):
        in_size = self.weights.shape[0]-1
        dx = np.dot(error_tensor, self.weights[:in_size].T)
        dW = np.dot(self.lastIn.T, error_tensor)
        if self.optimizer != None:
            self.weights = self.optimizer.calculate_update(self.weights, dW)      
        self.gradient_weights = dW
        return dx

