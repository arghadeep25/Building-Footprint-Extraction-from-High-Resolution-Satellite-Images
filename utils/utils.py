""" Name: Utils
    Author: Arghadeep Mazumder
    Version: 0.1
    Description: 1. File for generation of various data
                    - Inria Dataset
                    - DSAC Dataset
                    - CrowdAI Building Dataset
                    - Nucleus Dataset Kaggle
                 2.
"""
import numpy as np
import sys
import os
import cv2
import pandas as pd
from skimage.util import crop
from skimage.transform import resize
from skimage.io import imread, imshow, show
from sklearn.feature_extraction import image

class inria_data_generator():
    """ Class for generating Inria Building Dataset

        Original dataset: Image Size = 5000x5000
        Original Dataset: Mask Size  = 5000x5000

        Dataset Structure: AerialImageDataset
                            |_ test
                                |_ images
                            |_ train
                                |_images
                                |_gt

        Purpose of the class is to split the original dataset and
        corresponding mask into 250x250 pathces so that we can upscale
        the patches into size of 256x256 for training

        Parameters::
         - data_path: path for the train folder
           eg: (../AerialImageDataset/train/)
         - output_path: path to store the patches
         - patch_size: size of the patches (default: 250)

    """

    def __init__(self, data_path, output_path, patch_size = 250):
        self.data_path = data_path
        self.patch_size = patch_size
        self.output_path = output_path

    def load_image_mask(self, image_name, mask_name):
        """ Load an image from the folder
        """
        try:
            image = cv2.imread(image_name,3)
            mask = cv2.imread(mask_name,1)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            return image, mask
        except:
            print('Unable to read image or mask from the source folder')
            return

    def split_image(self, image_name, mask_name):
        """ Split an image into 250x250 patches and store into
            an array
        """
        image, mask = self.load_image_mask(image_name, mask_name)
        try:
            print('Split Image  :: Image Size: {}'.format(image.shape))
            print('Split Image  :: Mask Size:  {}'.format(mask.shape),'\n')
        except:
            print('Error Loading Mask...')
            return

        image_patches = []
        mask_patches = []

        image_patch = np.zeros((self.patch_size, self.patch_size, 3))
        mask_patch = np.zeros((self.patch_size, self.patch_size, 1))

        # Generating Image Patches
        for img_col in range(0, image.shape[0], self.patch_size):
            for img_row in range(0, image.shape[1], self.patch_size):
                image_patch = image[img_row : img_row + self.patch_size,
                                    img_col : img_col + self.patch_size]
                image_patches.append(image_patch)

        #  Generating Mask Patches
        for mask_col in range(0, mask.shape[0], self.patch_size):
            for mask_row in range(0, mask.shape[1], self.patch_size):
                mask_patch = mask[mask_row : mask_row + self.patch_size,
                                mask_col : mask_col + self.patch_size]
                mask_patches.append(mask_patch)

        return image_patches, mask_patches

    def save_image(self, image_patches, mask_patches, id_name):
        """ Save all the images and masks individually into given path
        """
        dir = os.path.join(self.output_path, 'inria_dataset_256/')
        output_dir = os.path.join(dir, 'train/')
        image_dir = os.path.join(output_dir, 'images/')
        mask_dir = os.path.join(output_dir, 'gt/')
        if not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        id_name, _ = os.path.splitext(id_name)

        for img in range(len(image_patches)):
            image_name =  image_dir + id_name + '_' + str(img) + '.tif'
            cv2.imwrite(image_name, image_patches[img])

        for mask in range(len(mask_patches)):
            mask_name = mask_dir + id_name + '_' + str(mask) + '.tif'
            cv2.imwrite(mask_name, mask_patches[mask])


    def split_all_images(self):
        """ Split all the images in the folder
        """
        image_path = os.path.join(self.data_path, 'images/')
        mask_path = os.path.join(self.data_path, 'gt/')
        _, _, files = next(os.walk(image_path))
        total_patches = 0
        for file in files:
            image_name = image_path + file
            mask_name = mask_path + file
            print('\nSpliting Image and Mask :: ', file,'\n')
            image_patches, mask_patches = self.split_image(image_name,
                                                            mask_name)
            self.save_image(image_patches, mask_patches, file)
            total_patches += len(image_patches)

        print('::Patch Summary::')
        print('Number of Image patches: ',total_patches)
        print('Size of Image Patch: ',image_patches[0].shape)
        print('Size of Mask Patch: ',mask_patches[0].shape)
