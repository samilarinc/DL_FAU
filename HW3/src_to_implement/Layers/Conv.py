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

    def forward(self, input_tensor):
        if input_tensor.ndim == 3:
            input_tensor = input_tensor[:, :, :, np.newaxis]
        self.lastShape = input_tensor.shape
        padded_image = np.zeros((input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2] + self.convolution_shape[1] - 1, input_tensor.shape[3] + self.convolution_shape[2] - 1))
        p1 = int(self.convolution_shape[1]//2 == self.convolution_shape[1]/2)
        p2 = int(self.convolution_shape[2]//2 == self.convolution_shape[2]/2)
        padded_image[:, :, (self.convolution_shape[1]//2):-(self.convolution_shape[1]//2)+p1, (self.convolution_shape[2]//2):-(self.convolution_shape[2]//2)+p2] = input_tensor
        input_tensor = padded_image
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
        
    def backward(self, error_tensor):

        # N = 1 # number of examples
        # F = 1 # number of filters
        # C = 1 # number of channels
        # H = 5 # height inputs
        # W = 5 # width inputs
        # HH = 3 # height filter
        # WW = 3 # width filter
        # x = np.random.randn(N, C, H, W)
        # w = np.random.randn(F, C, HH, WW)
        # b = np.random.randn(F,)
        # conv_param = {'stride': 1, 'pad': 0}

        # # stride = conv_param['stride']
        # pad = conv_param['pad']

        # stride = self.stride_shape

        # # dimensions of the output
        # H1 = int(1 + (H + 2 * pad - HH)/stride)
        # W1 = int(1 + (W + 2 * pad - WW)/stride)

        # # incoming gradient dL/dY
        # dout = np.random.randn(N, F, H1, W1)

        # dx = np.zeros(x.shape)
        # # loop through the number of examples
        # for n in range(N):
        #     # hi and wi - looping through x
        #     for hi in range(H):
        #         for wi in range(W):
        #             # i and j - looping through output 
        #             y_idxs = []
        #             w_idxs = []
        #             for i in range(H1):
        #                 for j in range(W1):
        #                     # check if within weights limits
        #                     if ((hi + pad - i * stride) >= 0) and ((hi + pad - i * stride) < HH) and ((wi + pad - j * stride) >= 0) and ((wi + pad - j * stride) < WW):
        #                         w_idxs.append((hi + pad - i * stride, wi + pad - j * stride))
        #                         y_idxs.append((i, j))

        #             # loop through filters
        #             for f in range(F):
        #                 dx[n, : , hi, wi] += np.sum([w[f, :, widx[0], widx[1]] * dout[n, f, yidx[0], yidx[1]] for widx, yidx in zip(w_idxs, y_idxs)], 0)

        # for f in range(F):
        #     # looping through channels
        #     for c in range(C):
        #         for i in range(HH):
        #             for j in range(WW):
        #                 dw[f, c, i ,j] = np.sum(padded_x[:,  c, i: i + H1 * stride : stride, j : j + W1* stride : stride] * dout[:, f, :, :])


        dx = np.dot(error_tensor, self.weights.T)
        dW = np.dot(self.lastIn.T, error_tensor)
        if self._optimizer != None:
            self.weights = self._optimizer.calculate_update(self.weights, dW)
            self.bias = self._optimizer.calculate_update(self.bias, error_tensor)
       
        self.gradient_bias = error_tensor
        self.gradient_weights = dW
       
        return dx




    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer
        self.bias = bias_initializer