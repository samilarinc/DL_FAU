import numpy as np
from scipy import signal
from Layers import Base
from scipy import convolve

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True
        if type(stride_shape) == int:
            stride_shape = (stride_shape, stride_shape)
        elif len(stride_shape) == 1:
            stride_shape = (stride_shape[0], stride_shape[0])
        self.stride_shape = stride_shape
        self.conv2d = (len(convolution_shape) == 3)
        if self.conv2d:
            self.convolution_shape = convolution_shape
        else:
            self.convolution_shape = (*convolution_shape, 1)
        self.num_kernels = num_kernels
        self.weights = np.random.uniform(size = (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(size = (num_kernels,))
        self.gradient_weights = None
        self.gradient_bias = None
        self._optimizer = None
        self.lastShape = None

    def forward(self, input_tensor):
        if input_tensor.ndim == 3:
            input_tensor = input_tensor[:, :, :, np.newaxis]
        self.lastShape = input_tensor.shape
        padded_image = np.zeros((input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2] + self.convolution_shape[1] - 1, input_tensor.shape[3] + self.convolution_shape[2] - 1))
        p1 = int(self.convolution_shape[1]//2 == self.convolution_shape[1]/2)
        p2 = int(self.convolution_shape[2]//2 == self.convolution_shape[2]/2)
        padded_image[:, :, (self.convolution_shape[1]//2):-(self.convolution_shape[1]//2)+p1, (self.convolution_shape[2]//2):-(self.convolution_shape[2]//2)+p2] = input_tensor
        input_tensor = padded_image
        self.padded = padded_image.copy()
        # dimensions of the output
        h_cnn = np.ceil((padded_image.shape[2] - self.convolution_shape[1] + 1) / self.stride_shape[0])
        v_cnn = np.ceil((padded_image.shape[3] - self.convolution_shape[2] + 1) / self.stride_shape[1])
            
        output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, int(h_cnn), int(v_cnn)))
        
        # loop through the number of examples
        for n in range(input_tensor.shape[0]):
            # loop through the number of filters
            for f in range(self.num_kernels):
                    # loop through the height of the output
                    for i in range(int(h_cnn)):
                        # loop through the width of the output
                        for j in range(int(v_cnn)):
                            # check if within weights limits
                            if ((i * self.stride_shape[0]) + self.convolution_shape[1] <= input_tensor.shape[2]) and ((j * self.stride_shape[1]) + self.convolution_shape[2] <= input_tensor.shape[3]):
                                output_tensor[n, f, i, j] = np.sum(input_tensor[n, :, i*self.stride_shape[0]:i*self.stride_shape[0] + self.convolution_shape[1], j * self.stride_shape[1]:j * self.stride_shape[1] + self.convolution_shape[2]] * self.weights[f, :, :, :])
                                output_tensor[n, f, i, j] += self.bias[f]
                            else:
                                output_tensor[n, f, i, j] = 0
        return output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weights = optimizer
        self._optimizer.bias = optimizer
        
    def backward(self, error_tensor):
        self.gradient_weights = np.zeros(self.weights.shape)
        self.gradient_bias = np.zeros(self.bias.shape)
        return_tensor = np.zeros(self.lastShape)
        padded_error = np.zeros((error_tensor.shape[0], error_tensor.shape[1], error_tensor.shape[2] + self.convolution_shape[1] - 1, error_tensor.shape[3] + self.convolution_shape[2] - 1))
        p1 = int(self.convolution_shape[1]//2 == self.convolution_shape[1]/2)
        p2 = int(self.convolution_shape[2]//2 == self.convolution_shape[2]/2)
        padded_error[:, :, (self.convolution_shape[1]//2):-(self.convolution_shape[1]//2)+p1, (self.convolution_shape[2]//2):-(self.convolution_shape[2]//2)+p2] = error_tensor
        # loop through the number of examples
        for n in range(padded_error.shape[0]):
            # loop through the number of filters
            for f in range(self.num_kernels):
                    # loop through the height of the output
                    for i in range(padded_error.shape[2]):
                        # loop through the width of the output
                        for j in range(padded_error.shape[3]):
                            # check if within weights limits
                            if ((i * self.stride_shape[0]) + self.convolution_shape[1] <= self.lastShape[2]) and ((j * self.stride_shape[1]) + self.convolution_shape[2] <= self.lastShape[3]):
                                return_tensor[n, :, i*self.stride_shape[0]:i*self.stride_shape[0] + self.convolution_shape[1], j*self.stride_shape[1]:j*self.stride_shape[1] + self.convolution_shape[2]] += error_tensor[n, f, i, j] * self.weights[f, :, :, :]
                                self.gradient_weights[f, :, :, :] += padded_error[n, f, i, j] * self.padded[n, :, i*self.stride_shape[0]:i*self.stride_shape[0] + self.convolution_shape[1], j*self.stride_shape[1]:j*self.stride_shape[1] + self.convolution_shape[2]]
                                self.gradient_bias[f] += padded_error[n, f, i, j]
                            else:
                                return_tensor[n, :, i*self.stride_shape[0]:i*self.stride_shape[0] + self.convolution_shape[1], j*self.stride_shape[1]:j*self.stride_shape[1] + self.convolution_shape[2]] += 0
                                self.gradient_weights[f, :, :, :] += 0
                                self.gradient_bias[f] += 0
                                
        if self.optimizer is not None:
            self.weights = self._optimizer.weights.update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.bias.update(self.bias, self.gradient_bias)

        return return_tensor

        
    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer
        self.bias = bias_initializer