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
import keras.utils
from .augmentation import DataAugmentation
from keras.preprocessing.image import ImageDataGenerator

class InriaDataLoader(keras.utils.Sequence):
    """ Load the training dataset for Inria Dataset from the
        data folder and put in an array

        Parameters: - data_path: Loads the datapath
                    - patch_size: Format the image sizes (default: 256x256)
                    - train_ids:

    """
    def __init__(self, data_ids, data_path, patch_size = 256,
                batch_size = 8, aug = False, rotation = 0,
                zoom_range = 1, horizontal_flip = False,
                vertical_flip = False, shear = 0):

        self.data_path = data_path
        self.data_ids = data_ids
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.aug = aug
        self.rotation = rotation
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.shear = shear

    def __load__(self, data_name):
        """ Load an image and a mask from the data folder
            Parameters: Image name
        """
        image_name_path = os.path.join(self.data_path,'images/', data_name)
        mask_name_path = os.path.join(self.data_path, 'gt', data_name)

        image = cv2.imread(image_name_path)
        image = cv2.resize(image, (self.patch_size, self.patch_size))

        mask = cv2.imread(mask_name_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (self.patch_size, self.patch_size))
        mask = mask[:, :, np.newaxis]

        image = image/255.
        mask = mask/255.

        return image, mask

    def __getitem__(self, index):
        """ Get all the images and masks in the data folder
            and put into array
        """
        if(index+1)*self.batch_size > len(self.data_ids):
            self.batch_size = len(self.data_ids) - index*self.batch_size

        files_batch = self.data_ids[index * \
            self.batch_size: (index + 1) * self.batch_size]

        images = []
        masks = []

        for file in files_batch:
            image, mask = self.__load__(file)
            aug = DataAugmentation(image, mask,
                                rotation =self.rotation,
                                zoom_range = self.zoom_range,
                                activate = self.aug)

            aug_images, aug_masks = aug.augment()

            for aug_image in aug_images:
                images.append(aug_image)

            for aug_mask in aug_masks:
                masks.append(aug_mask)

        images = np.array(images)
        masks = np.array(masks)

        return images, masks


    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.data_ids)/float(self.batch_size)))
