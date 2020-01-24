import numpy as np

class Linear():
    def __init__(self):
        self.val = np.linspace(0, 10, 1000)

    def function(self):
        x = self.val + 4
        return self.val, x

    def derivative(self):
        der = 0*self.val + 4
        return der, self.val
