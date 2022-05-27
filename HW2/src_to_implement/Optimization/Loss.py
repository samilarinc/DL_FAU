import numpy as np

class CrossEntropyLoss(object):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, prediction_tensor, label_tensor):
        print("predtensor",prediction_tensor)
        # self.previn = label_tensor
        loss = -np.sum((label_tensor)*(np.log(prediction_tensor)))
        return loss/float(prediction_tensor.shape[0])
    
    def backward(self, label_tensor):
        return label_tensor