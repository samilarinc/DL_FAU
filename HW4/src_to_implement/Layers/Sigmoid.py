import numpy as np
from Layers import Base

class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return 1 / (1 + np.exp(-input_tensor))

    def backward(self, error_tensor):
        temp = self.forward(error_tensor)
        return temp * (1 - temp)