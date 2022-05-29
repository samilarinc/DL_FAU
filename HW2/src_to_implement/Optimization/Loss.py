import sys
sys.path.append('../')

import numpy as np
from Layers.SoftMax import SoftMax

class CrossEntropyLoss(SoftMax):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, prediction_tensor, label_tensor):
        # print("predtensor",prediction_tensor)
        # self.previn = label_tensor
        inp = SoftMax.forward(self,prediction_tensor)
        loss = -np.sum(label_tensor*np.log(inp))
        # loss = -(label_tensor*np.log(prediction_tensor)+(1-label_tensor)*np.log(1-prediction_tensor))
        
        # return loss/float(prediction_tensor.shape[0])
        return loss
    
    def backward(self, label_tensor):
        return label_tensor
    
    
label_tensor = np.zeros((9, 4))
label_tensor[:, 2] = 1
input_tensor = np.zeros_like(label_tensor)
input_tensor[:, 1] = 1
# input_tensor[:,:] = 0.4
layer = CrossEntropyLoss()
loss = layer.forward(input_tensor, label_tensor)
loss2 = layer.forward(label_tensor, label_tensor)



# np.log(label_tensor)