import numpy as np

class LeakyRelu():
    def __init__(self):
        self.val = np.linspace(-5, 5, 200)

    def function(self):
        return np.maximum(0.01*self.val, self.val), self.val

    def derivative(self):
        der = np.ones_like(self.val)
        der[self.val < 0] = 0.01
        return der, self.val
