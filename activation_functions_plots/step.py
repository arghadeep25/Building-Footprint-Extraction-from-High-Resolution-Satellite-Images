import numpy as np

class Step():
    def __init__(self):
        self.val = np.arange(-5, 5, .02)
    def function(self):
        z = self.val
        step_fn = np.vectorize(lambda z: 1.0 if z >= 0.0 else 0.0)
        step = step_fn(self.val)
        return step, self.val

    def derivative(self):
        z = np.zeros([len(self.val), 1])
        return z, self.val
