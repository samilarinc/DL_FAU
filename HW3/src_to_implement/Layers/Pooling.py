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
        self.x_s = np.zeros((*input_tensor.shape[0:2], int(h_pools), int(v_pools)), dtype=int)
        self.y_s = np.zeros((*input_tensor.shape[0:2], int(h_pools), int(v_pools)), dtype=int)
        
        a = -1
        for i in range(0, input_tensor.shape[2] - self.pooling_shape[0] + 1, self.stride_shape[0]):
            a += 1
            b = -1
            for j in range(0, input_tensor.shape[3] - self.pooling_shape[1] + 1, self.stride_shape[1]):
                b += 1
                temp = input_tensor[:, :, i:i+self.pooling_shape[0], j:j+self.pooling_shape[1]].reshape(*input_tensor.shape[0:2], -1)
                output_pos = np.argmax(temp, axis = 2)
                x = output_pos // self.pooling_shape[1]
                y = output_pos % self.pooling_shape[1]
                print(x.shape)
                self.x_s[:, :, a, b] = x
                self.y_s[:, :, a, b] = y
                # print(output_pos.shape == input_tensor.shape[0:2])
                output_tensor[:, :, a, b] = np.choose(output_pos, np.moveaxis(temp, 2, 0))         #np.max(temp, axis = 2)
                #np.max(input_tensor[:, :, i:i+self.pooling_shape[0], j:j+self.pooling_shape[1]], axis =(2, 3))
                
        return output_tensor
    
    def backward(self, error_tensor):
        return_tensor = np.zeros_like(self.lastIn)
        print(self.x_s)
        import time
        time.sleep(10)
        for i in range(self.x_s.shape[2]):
            for j in range(self.y_s.shape[3]):
                return_tensor[:, :, i*self.stride_shape[0]+self.x_s[:, :, i, j], j*self.stride_shape[1]+self.y_s[:, :, i, j]] = error_tensor[:, :, i, j]
        return return_tensor