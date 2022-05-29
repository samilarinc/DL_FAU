import numpy as np
from Optimization import Optimizers
from Layers import Base

class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.lastIn = input_tensor
        temp = np.exp(input_tensor - input_tensor.max())
        self.lastOut = temp / temp.sum()
        return self.lastOut

    def backward(self, error_tensor):
        return self._optimizer.calculate_update(self.lastIn, error_tensor)