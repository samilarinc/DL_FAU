import copy


class NeuralNetwork(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None
    
    def forward(self):
        raise NotImplementedError
    
    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)
    
    def train(self, iterations):
        raise NotImplementedError

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer(input_tensor)
        return input_tensor