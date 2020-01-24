"""
Name: Data Loader
Author: Arghadeep Mazumder
Version: 0.1
Description: 1. File for generation of various data
                - Inria Training Dataset
                - Inria Test Dataset
                - CrowdAI Building Dataset
                - SpaceNet Buidling Dataset
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
sys.path.append('../')
from keras.preprocessing.image import ImageDataGenerator, img_to_array


class InriaDataLoader(keras.utils.Sequence):
    """ Load the training dataset for Inria Dataset from the
        data folder and put in an array

        Parameters: - data_path: Loads the datapath
                    - patch_size: Format the image sizes (default: 256x256)
                    - train_ids:

    """
    def __init__(self, data_ids, data_path, patch_size = 256,
                batch_size = 8, aug = True, rotation = 90,
                zoom_range = 1.5, horizontal_flip = True, hist_eq = True,
                vertical_flip = True, shear = 0.2, brightness = True,
                add_noise = True, sigma = 10, split_channel = False):

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
        self.split_channel = split_channel
        self.hist_eq = hist_eq
        self.brightness = brightness
        self.add_noise = add_noise
        self.sigma = sigma

    def __load__(self, data_name):
        """ Load an image and a mask from the data folder
            Parameters: Image name
        """
        image_name_path = os.path.join(self.data_path,'images/', data_name)
        mask_name_path = os.path.join(self.data_path, 'gt', data_name)

        image = cv2.imread(image_name_path)
        image = img_to_array(image)

        mask = cv2.imread(mask_name_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # PSPNet and SegNet
        if self.split_channel == True:
            nclasses = len(np.unique(mask))
            split_mask = np.zeros((self.patch_size, self.patch_size, 2))
            for c in range(nclasses):
                split_mask[:, :, c] = (mask == c).astype(int)
            split_mask = cv2.resize(split_mask,
                                    (self.patch_size, self.patch_size),
                                    interpolation=cv2.INTER_NEAREST)
            split_mask = np.reshape(split_mask,
                                    (self.patch_size*self.patch_size, 2))
            return image, split_mask

        mask = mask[:, :, np.newaxis]

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
                                   horizontal_flip = self.horizontal_flip,
                                   vertical_flip = self.vertical_flip,
                                   shear = self.shear,
                                   hist_eq = self.hist_eq,
                                   brightness = self.brightness,
                                   add_noise = self.add_noise,
                                   sigma = self.sigma,
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
