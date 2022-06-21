import numpy as np
from Layers import Pooling as P

stride_shape = (1, 1)
pooling_shape = (2, 2)

layer = P.Pooling(stride_shape, pooling_shape)
input_tensor = [[9, 8, 0, 5], [3, 5, 1, 1], [1, 1, 6, 3], [5, 2, 6, 3]]
input_tensor = np.array(input_tensor)
input_tensor = input_tensor[np.newaxis, np.newaxis, :, :]

error_tensor1 = [[6, 3], [7, 2]]
error_tensor2 = [[6,4,3], [2,5,4] , [7,1,2]]
error_tensor = np.array(error_tensor2)
error_tensor = error_tensor[np.newaxis, np.newaxis, :, :]

layer.forward(input_tensor)
print(layer.backward(error_tensor))