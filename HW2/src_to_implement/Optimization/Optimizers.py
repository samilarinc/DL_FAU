class Sgd(object):
    def __init__(self, learning_rate:float):
        self.lr = learning_rate
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - gradient_tensor * self.lr 
        return weight_tensor