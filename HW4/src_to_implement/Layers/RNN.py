import numpy as np
from Layers import Base

class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self._memorize = False
        self.hidden_state = np.zeros((1, hidden_size))

    def forward(self, X):
        raise NotImplementedError
        if _memorize:
            self.hidden_state = X
        self.X = X
        self.Y = np.zeros_like(X)
        for i in range(X.shape[0]):
            self.Y[i] = self.hidden_state
            self.hidden_state = self.hidden_state + X[i]
        return self.Y

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value