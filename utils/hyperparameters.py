"""
Name: Hyperparameters
Author: Arghadeep Mazumder
Version: 0.1
Description: 1. U-Net Hyperparameters
                - image_size (default = 256)
                - train_path
                - epochs (default = 10)
                - batch_size = 8
                - train_ids

             2. Mask R-CNN Hyperparameters
                -
                -
                -
                -

"""
import numpy as np
import os
import sys

class HyperparameterUNet():
    """ Class for generating U-Net Hyperparameters
        Parameters: - image_size (default = 256)
                    - train_path
                    - epochs (default = 10)
                    - batch_size = 8
                    - train_ids (Number of Files in the folder)
    """
    def __init__(self,
                image_size = 256,
                train_path,
                epochs = 10,
                batch_size = 8,
                train_ids):

        self.image_size = image_size
        self.train_path = train_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_ids = train_ids

    def params(self):
        image_size = self.image_size
        train_path = self.train_path
        epochs = self.epochs
        batch_size = self.batch_size

        # Extracting the number of files in the train folder
        image_path = os.path.join(train_path,'images/')
        _, _, image_names = next(os.walk(image_path))
        train_ids = len(image_names)


        return image_size, train_path, epochs, batch_size, train_ids

class HyperparameterMaskRCNN():
    """ Class for generating Mask R-CNN Hyperparameters
        Parameters: -
                    -
    """
    def __int__(self, iamge_size):
        self.image_size = iamge_size
    def params(self):
        image_size = self.image_size
        return iamge_size
