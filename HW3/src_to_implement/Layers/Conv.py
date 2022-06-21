import numpy as np
from scipy import signal
from Layers import Base
from scipy import convolve
from scipy.signal import correlate2d, convolve2d

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True
        self.conv_shape = convolution_shape
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
        self.stride_row = self.stride_shape[0]
        self.conv_row = convolution_shape[1]
        if len(convolution_shape) == 3:  # for 3d convolution
            self.weights = np.random.rand(num_kernels, *convolution_shape)
            self.bias = np.random.rand(num_kernels)  # 1 bias per kernel
            self.stride_col = self.stride_shape[1]
            self.conv_col = convolution_shape[2]
            self.dim1 = False
        else:  # distinction for 2d convlution
            self.weights = np.random.rand(num_kernels, convolution_shape[0], convolution_shape[1], 1)
            self.bias = np.random.rand(num_kernels)
            self.stride_col = 1
            self.conv_col = 1
            self.dim1 = True  # boolean for the 2d case

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
        self.input_tensor = input_tensor
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
        self.output_shape = np.shape(output_tensor)

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

        ######## Initilize
        self.error_T = error_tensor.reshape(self.output_shape)

        # upsampling
        self.up_error_T = np.zeros((self.input_tensor.shape[0], self.num_kernels, *self.input_tensor.shape[2:]))  # num_ker=num chanels
        next_error = np.zeros(self.input_tensor.shape)  # we have the same size of the input
        # For Padded input image
        self.padding_X = np.zeros((*self.input_tensor.shape[:2], self.input_tensor.shape[2] + self.conv_row - 1,
                                   self.input_tensor.shape[3] + self.conv_col - 1))
        # Bias
        self.grad_bias = np.zeros(self.num_kernels)
        # gradient with respect to the weights
        self.grad_weights = np.zeros(self.weights.shape)

        #########################

        # Padding
        # input padding we pad with half of the kernel size
        pad_up = int(np.floor(self.conv_col / 2))  # (3, 5, 8)
        pad_left = int(np.floor(self.conv_row / 2))

        for batch in range(self.up_error_T.shape[0]):
            for ker_num in range(self.up_error_T.shape[1]):
                # gradient with respect to the bias
                self.grad_bias[ker_num] += np.sum(error_tensor[batch, ker_num, :])

                for ht in range(self.error_T.shape[2]):
                    for wdt in range(self.error_T.shape[3]):
                        self.up_error_T[batch, ker_num, ht * self.stride_row, wdt * self.stride_col] = self.error_T[
                            batch, ker_num, ht, wdt]  # we fill up with the strided error tensor

                for ch in range(self.input_tensor.shape[1]):  # channel num
                    next_error[batch, ch, :] += convolve2d(self.up_error_T[batch, ker_num, :], self.weights[ker_num, ch, :],'same')  # same

            # Referred from
            for num in range(self.input_tensor.shape[1]):
                for ht in range(self.padding_X.shape[2]):
                    for wdt in range(self.padding_X.shape[3]):
                        if (ht > pad_left - 1) and (ht < self.input_tensor.shape[2] + pad_left):
                            if (wdt > pad_up - 1) and (wdt < self.input_tensor.shape[3] + pad_up):
                                self.padding_X[batch, num, ht, wdt] = self.input_tensor[batch, num, ht - pad_left, wdt - pad_up]

            for ker_num in range(self.num_kernels):
                for ch in range(self.input_tensor.shape[1]):
                    self.grad_weights[ker_num, ch, :] += correlate2d(self.padding_X[batch, ch, :],
                                                              self.up_error_T[batch, ker_num, :],'valid')  # convolution of the error tensor with the padded input tensor

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.grad_weights)

        if self._optimizer is not None:
            self.bias = self._optimizer.calculate_update(self.bias, self.grad_bias)

        # again distinction between 2d and 3d
        if len(self.conv_shape) == 2:   #if self.dim1:
            next_error = next_error.reshape(next_error.shape[0],next_error.shape[1], next_error.shape[2])

        return next_error

        
    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer
        self.bias = bias_initializer