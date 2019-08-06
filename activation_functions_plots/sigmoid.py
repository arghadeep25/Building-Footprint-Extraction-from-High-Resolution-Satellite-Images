import numpy as np

class Sigmoid():
    def __init__(self):
        self.val = np.arange(-5, 5, .1)

    def function(self):
        z = self.val
        sigma_fn = np.vectorize(lambda z: 1/(1+np.exp(-z)))
        sigma = sigma_fn(z)
        return sigma, self.val

    def derivative(self):
        z = self.val
        sigma_fn = np.vectorize(lambda z: 1/(1+np.exp(-z)))
        sigma = sigma_fn(z)
        sigma_der = sigma*(1 - sigma)
        return sigma_der, self.val
