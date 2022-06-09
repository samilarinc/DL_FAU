import numpy as np
from Layers import Base

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.uniform(size = (num_kernels, num_kernels, *convolution_shape))
        self.bias = np.random.uniform(size = (convolution_shape))
        self.gradient_weights = None
        self.gradient_bias = None

    def forward(self, input_tensor):
        raise NotImplementedError



    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer
        self.bias = bias_initializer