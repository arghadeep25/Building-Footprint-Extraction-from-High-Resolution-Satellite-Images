"""
Name: Data Loader
Author: Arghadeep Mazumder
Version: 0.1
Description: 1. File for generation of various data
                - Inria Dataset
                - DSAC Dataset
                - CrowdAI Building Dataset
                - Nucleus Dataset Kaggle
             2.
"""
import os
import sys
import cv2
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize

class InriaDataLoader(keras.utils.Sequence):
    """ Load the training dataset for Inria Dataset

        Data structure: |_ train/
                            |_images
                                |_...
                            |_gt
                                |_...

        Parameters: - data_path: Loads the datapath
                    - patch_size: Format the image sizes (default: 256x256)

    """
    def __init__(self,  data_path, patch_size = 256):
        self.data_path = data_path
        self.patch_size = patch_size

    def __load__(self, image_name, mask_name):
        """ Load an image and a mask from the data folder
        """
        image = cv2.imread(image_name)
        image = resize(image, (self.patch_size, self.patch_size))
        mask = cv2.imread(mask_name)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = resize(mask, (self.patch_size, self.patch_size))
        return image, mask

    def __getitem__(self):
        """ Get all the images and masks in the data folder
            and put into array
        """
        images = []
        masks = []
        image_path = os.path.join(self.data_path, 'images/')
        mask_path = os.path.join(self.data_path, 'gt/')
        _, _, files = next(os.walk(image_path))
        total_patches = 0
        for file in files:
            image_name = image_path + file
            mask_name = mask_path + file
            image, mask = self.__load__(image_name, mask_name)
            images.append(image)
            masks.append(mask)

        return images, masks

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))
