import numpy as np

class ReLU():
    def __init__(self):
        self.val = np.arange(-20, 20, .1)

    def function(self):
        zero = np.zeros(len(self.val))
        y = np.max([zero, self.val], axis=0)
        return y, self.val

    def derivative(self):
        val_func, _ = self.function()
        val_func[val_func<=0.0] = 0
        val_func[val_func>0.0] = 1
        return val_func, self.val
