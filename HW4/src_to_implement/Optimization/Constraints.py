import numpy as np

class L2_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def norm(self, weight_tensor):
        return np.sum(np.square(weight_tensor))
    
    def calculate_gradient(self, weight_tensor):
        return 2 * self.alpha * weight_tensor

class L1_Regularizer(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def norm(self, weight_tensor):
        return np.sum(np.abs(weight_tensor))
    
    def calculate_gradient(self, weight_tensor):
        return np.sign(weight_tensor) * self.alpha
