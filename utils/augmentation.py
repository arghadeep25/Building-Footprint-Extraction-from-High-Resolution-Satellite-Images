"""
Name: Data Augmentation
Author: Arghadeep Mazumder
Version: 0.1
Description: 1. Rotate
             2. Rescale
             3. Flip Horizontally
             4. Flip Vertically
             5. Shear
"""
import cv2
import numpy as np
from skimage.transform import resize, rotate, rescale, warp, AffineTransform

class DataAugmentation():
    """ Class for data augmentation.

        Mainly helps when the training data is small

        Parameters: image, mask,
                    rotation angle (default = 90)
                    zoom_range (default = 1)
                    horizontal_flip (default = False)
                    vertical_flip (default = False)
                    activate (default = False)
    """

    def __init__(self, image, mask,
                rotation = 0,
                zoom_range = 1,
                horizontal_flip = False,
                vertical_flip = False,
                shear = 0,
                activate = False):

        self.image = image
        self.mask = mask
        self.rotation = rotation
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.shear = shear
        self.activate = activate

    def rotate_data(self):
        """ Rotation
        """
        self.image = rotate(self.image, self.rotation)
        self.mask = rotate(self.mask, self.rotation)
        return self.image, self.mask

    def rescale_data(self):
        """ Rescaling
        """
        height, width, _ = self.image.shape
        self.image = rescale(self.image, self.zoom_range)
        self.image = resize(self.image, (height, width))

        self.mask = rescale(self.mask, self.zoom_range)
        self.mask = resize(self.mask, (height, width))
        return self.image, self.mask

    def flip_horizontal_data(self):
        """ Flip Horizontally
        """
        if self.flip_horizontal == True:
            flipped_image = np.flip(self.image, 1)
            flipped_mask = np.flip(self.mask, 1)
            return flipped_image, flipped_mask

    def flip_vertically_data(self):
        """ Flip Vertically
        """
        if self.flip_vertically == True:
            flipped_image = np.flip(self.image, 0)
            flipped_mask = np.flip(self.mask, 0)
            return flipped_image, flipped_mask

    def shear_data(self):
        """ Shear
        """
        trans = AffineTransform(shear = 0.2)
        self.image = warp(self.image, inverse_map= trans)
        self.mask = warp(self.mask, inverse_map= trans)
        return self.image, self.mask

    def augment(self):
        if self.activate == True:
            images = []
            masks = []
            images.append(self.image)
            masks.append(self.mask)
            # print('Augmentation:: Image List Size: ',len(images))
            if self.rotation != 0:
                self.image, self.mask = self.rotate_data()
                images.append(self.image)
                masks.append(self.mask)

            if self.zoom_range != 1:
                self.image, self.mask = self.rescale_data()
                images.append(self.image)
                masks.append(self.mask)

            if self.horizontal_flip == True:
                self.image, self.mask = self.flip_horizontal_data()
                images.append(self.image)
                masks.append(self.mask)

            if self.vertical_flip == True:
                self.image, self.mask = self.flip_vertically_data()
                images.append(self.image)
                masks.append(self.mask)

            if self.shear != 0:
                self.image, self.mask = self.shear_data()
                images.append(self.image)
                masks.append(self.mask)
        else:
            images = []
            masks = []
            images.append(self.image)
            masks.append(self.mask)

        images = np.array(images)
        masks = np.array(masks)

        return images, masks
