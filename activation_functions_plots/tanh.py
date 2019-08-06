import numpy as np

class TanH():
    def __init__(self):
        self.val = np.arange(-5, 5, .1)

    def function(self):
        return (np.tanh(self.val)), self.val

    def derivative(self):
        vals, _ = self.function()
        return (1.0 - np.tanh(self.val)**2), self.val
