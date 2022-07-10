import numpy as np
from Layers import Base

class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return np.tanh(input_tensor)

    def backward(self, error_tensor):
        return 1 - np.square(np.tanh(error_tensor))
