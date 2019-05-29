"""
Name: Visualizer
Author: Arghadeep Mazumder
Version: 0.1
Description: 1. Visualizer
             2.
"""
import numpy as np
import matplotlib.pyplot as plt

class InriaVisualizer():
    """ Class for visualizing input image and corresponding mask

        Parameters: Image
                    Mask
                    Image Size (default = 256)
    """
    def __init__(self, image, mask, image_size = 256):
        self.image = image
        self.mask = mask
        self.image_size = image_size

    def plot(self):
        fig = plt.figure()
        fig.subplots_adjust(hspace = 0.4, wspace = 0.4)

        fig_a = fig.add_subplot(1, 2, 1)
        fig_a.set_title('Input Image')
        image = np.reshape(self.image[0]*255,
                        (self.image_size, self.image_size))
        plt.imshow(image)

        fig_b = fig.add_subplot(1, 2, 2)
        fig_b.set_title('Output Mask')
        mask = np.reshape(self.mask[0]*255,
                        (self.image_size, self.image_size))
        plt.imshow(mask)

        plt.show()
