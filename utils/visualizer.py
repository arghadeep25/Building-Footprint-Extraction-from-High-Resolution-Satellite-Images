"""
Name: Visualizer
Author: Arghadeep Mazumder
Version: 0.1
Description: 1. Visualizer
             2.
"""
import numpy as np
import matplotlib.pyplot as plt

class Visualizer():
    """ Class for visualizing input image and corresponding
        mask

        Parameters: image
                    mask
    """
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask

    def plot(self):
        fig = plt.figure()

        fig_a = fig.add_subplot(1, 2, 1)
        a.set_title('Input Image')
        plt.imshow(image)

        fig_b = fig.add_subplot(1, 2, 2)
        b.set_title('Output Mask')
        plt.imshow(mask)
        plt.show()
