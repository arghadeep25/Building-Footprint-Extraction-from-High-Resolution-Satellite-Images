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
from .augmentation import DataAugmentation

class InriaDataLoader(keras.utils.Sequence):
    """ Load the training dataset for Inria Dataset from the
        data folder and put in an array

        Parameters: - data_path: Loads the datapath
                    - patch_size: Format the image sizes (default: 256x256)
                    - train_ids:

    """
    def __init__(self, data_path, patch_size = 256,
                aug = False, rotation = 0,
                zoom_range = 1, horizontal_flip = False,
                vertical_flip = False, shear = 0):

        self.data_path = data_path
        self.patch_size = patch_size
        self.aug= aug
        self.rotation = rotation
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.shear = shear
        # self.train_ids = train_ids

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
            aug = DataAugmentation(image, mask,
                                rotation =self.rotation,
                                zoom_range = self.zoom_range,
                                activate = self.aug)
            aug_images, aug_masks = aug.augment()
            for aug_image in aug_images:
                images.append(aug_image)
            for aug_mask in aug_masks:
                masks.append(aug_mask)
            # images.append(image)
            # masks.append(mask)

        return images, masks

    def on_epoch_end(self):
        pass

    def __len__(self):
        image_path = os.path.join(self.data_path, 'images/')
        _, _, files = next(os.walk(image_path))
        return int(np.ceil(len(files)/float(self.batch_size)))
