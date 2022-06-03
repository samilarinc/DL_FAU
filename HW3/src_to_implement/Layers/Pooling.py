import numpy as np
from Layers import Base

class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.lastIn = input_tensor
        h_pools = np.ceil((input_tensor.shape[2] - self.pooling_shape[0] + 1) / self.stride_shape[0])
        v_pools = np.ceil((input_tensor.shape[3] - self.pooling_shape[1] + 1) / self.stride_shape[1])
        output_tensor = np.zeros((*input_tensor.shape[0:2], int(h_pools), int(v_pools)))
        a = -1
        for i in range(0, input_tensor.shape[2] - self.pooling_shape[0] + 1, self.stride_shape[0]):
            a += 1
            b = -1
            for j in range(0, input_tensor.shape[3] - self.pooling_shape[1] + 1, self.stride_shape[1]):
                b += 1
                output_tensor[:, :, a, b] = np.max(input_tensor[:, :, i:i+self.pooling_shape[0], j:j+self.pooling_shape[1]], axis =(2, 3))
        return output_tensor
    
    def backward(self, error_tensor):
        raise NotImplementedError