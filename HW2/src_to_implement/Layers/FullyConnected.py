import numpy as np
from Optimization import Optimizers

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.radom.uniform(size = (output_size, input_size))
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = Optimizers.Sgd()

    def forward(input_tensor):
        return np.dot(self.weights, input_tensor)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def x(self, opt):
        self._optimizer = opt
     
    def backward(error_tensor):
        