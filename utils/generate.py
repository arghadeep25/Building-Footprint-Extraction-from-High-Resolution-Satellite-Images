"""
Name: Data Generator
Author: Arghadeep Mazumder
Version: 0.1
Description: 1. File for generation of various data
                - Inria Dataset
                - Inria Test Dataset
                - CrowdAI Building Dataset
                - SpaceNet Building Dataset
             2.
"""
import os
import sys
import cv2
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from osgeo import gdal
from scipy import ndimage
from skimage.util import crop
from pycocotools.coco import COCO
from skimage.transform import resize
from pycocotools import mask as cocomask
from skimage.io import imread, imshow, show
from sklearn.feature_extraction import image

class InriaDataGenerator:
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
        the patches into size of 256x256 for training.

        # Note: Run only "split_all_images" funciton.

        Parameters::
         - data_path: path for the train folder
           eg: (../AerialImageDataset/train/)
         - output_path: path to store the patches
         - patch_size: size of the patches (default: 256)

    """

    def __init__(self, data_path, output_path, patch_size = 256):
        self.data_path = data_path
        self.patch_size = patch_size
        self.output_path = output_path

    def load_image_mask(self, image_name, mask_name):
        """ Load image and mask from the folder
        """
        try:
            image = cv2.imread(image_name,3)
            mask = cv2.imread(mask_name,1)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            height, width, channel = image.shape
            num_patches = height//self.patch_size
            crop_dim = num_patches*self.patch_size

            return image[0:crop_dim, 0:crop_dim], mask[0:crop_dim, 0:crop_dim]
        except:
            print('Unable to read image or mask from the source folder')
            return

    def split_image(self, image_name, mask_name):
        """ Split image and mask into 256x256 patches and store into
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
                if image_patch.shape[0] == self.patch_size and image_patch.shape[1] == self.patch_size:
                    image_patches.append(image_patch)

        #  Generating Mask Patches
        for mask_col in range(0, mask.shape[0], self.patch_size):
            for mask_row in range(0, mask.shape[1], self.patch_size):
                mask_patch = mask[mask_row : mask_row + self.patch_size,
                                mask_col : mask_col + self.patch_size]
                if mask_patch.shape[0] == self.patch_size and mask_patch.shape[1] == self.patch_size:
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
            image_name =  image_dir + id_name + '_' + str(img) + '.png'
            cv2.imwrite(image_name, image_patches[img])

        for mask in range(len(mask_patches)):
            mask_name = mask_dir + id_name + '_' + str(mask) + '.png'
            cv2.imwrite(mask_name, mask_patches[mask])


    def split_all_images(self):
        """ Split all the images and masks in the folder
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

class InriaTestDataGenerator:
    """
        Class for generating Inria Test Dataset

        Dataset Structure: AerialImageDataset
                            |_ test
                                |_ images
    """
    def __init__(self, data_path, output_path, patch_size = 256):
        self.data_path = data_path
        self.output_path = output_path
        self.patch_size = patch_size

    def load_image(self, image_name):
        image = cv2.imread(image_name,3)
        height, width, channel = image.shape
        num_patches = height//self.patch_size
        crop_dim = num_patches*self.patch_size

        return image[0:crop_dim, 0:crop_dim]

    def split_image(self, image_name):
        """ Split image into patches and store into an array
        """
        image = self.load_image(image_name)
        try:
            print('Split Image  :: Image Size: {}'.format(image.shape),'\n')
        except:
            print('Error Loading Mask...')
            return

        image_patches = []

        image_patch = np.zeros((self.patch_size, self.patch_size, 3))

        # Generating Image Patches
        for img_col in range(0, image.shape[0], self.patch_size):
            for img_row in range(0, image.shape[1], self.patch_size):
                image_patch = image[img_row : img_row + self.patch_size,
                                    img_col : img_col + self.patch_size]
                image_patches.append(image_patch)

        return image_patches


    def save_image(self, image_patches, id_name):
        """ Save all the images and masks individually into given path
        """
        dir = os.path.join(self.output_path, 'inria_test_data_384/')
        output_dir = os.path.join(dir, 'test/')
        image_dir = os.path.join(output_dir, 'images/')
        if not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        id_name, _ = os.path.splitext(id_name)

        for img in range(len(image_patches)):
            image_name =  image_dir + id_name + '_' + str(img) + '.tif'
            cv2.imwrite(image_name, image_patches[img])


    def split_all_images(self):
        """ Split all the images and masks in the folder
        """
        image_path = os.path.join(self.data_path, 'images/')
        _, _, files = next(os.walk(image_path))
        total_patches = 0
        for file in files:
            image_name = image_path + file
            print('\nSpliting Images :: ', file,'\n')
            image_patches = self.split_image(image_name)
            self.save_image(image_patches, file)
            total_patches += len(image_patches)

        print('::Patch Summary::')
        print('Number of Image patches: ',total_patches)
        print('Size of Image Patch: ',image_patches[0].shape)

class CrowdAIDataGenerator:
    """ Class for generating CrowdAI dataset

        Dataset Structure: CrowdAI Mapping Challenge
                            |_ test/
                            |_ train/
                                |_images/
                                |_annotation.json
                                |_annotation-small.json
                            |_val/
                                |_images/
                                |_annotation.json
                                |_annotation-small.json
    """
    def __init__(self, data_dir, out_dir, patch_size = 256, gen_type='small'):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.patch_size = patch_size
        self.gen_type = gen_type
        self.train_image_dir = os.path.join(self.data_dir,
                                            'train/images/')
        self.train_ann_path = os.path.join(self.data_dir,
                                           'train/annotation.json')
        self.train_ann_small_path = os.path.join(self.data_dir,
                                            'train/annotation-small.json')

        self.val_image_dir = os.path.join(self.data_dir,
                                          'val/images')
        self.val_ann_path = os.path.join(self.dir,
                                         'val/annotation.json')
        self.val_ann_small_path = os.path.join(self.data_dir,
                                          'val/annotation-small.json')

    def generate(self):
        """Generate binary mask from annotation file
        """
        # Loading annotation path into memory
        coco = COCO(train_annotation_small_path)
        category_ids = coco.loadCats(coco.getCatIds())
        # Generating lists of all images
        image_ids = coco.getImgIds(catIds=coco.getCatIds())

        for image_id in image_ids:
            img = coco.loadImgs(image_id)[0]
            image_name = self.out_dir + 'images/' + str(image_id) + '.png'
            mask_name = self.out_dir + 'gt/' + str(image_id) + '.png'
            image_path = os.path.join(self.train_images_dir, img['file_name'])
            I = cv2.imread(image_path)
            annotation_ids = coco.getAnnIds(imgIds=img['id'])
            annotations = coco.loadAnns(annotation_ids)
            mask = np.zeros((300, 300))
            for _idx, annotation in enumerate(annotations):
                rle = cocomask.frPyObjects(annotation['segmentation'],
                                           img['height'],
                                           img['width'])
                m = cocomask.decode(rle)
                m = m.reshape((img['height'], img['width']))
                mask = np.maximum(mask, m)

            resized_img = cv2.resize(I, (self.patch_size,
                                         self.patch_size),
                                     interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(image_name, resized_img)

            resized_mask = cv2.resize(mask, (self.patch_size,
                                             self.patch_size),
                                      cv2.INTER_NEAREST)

            for i in range(resized_mask.shape[0]):
                for j in range(resized_mask.shape[1]):
                    if resized_mask[i,j] <= 70:
                        resized_mask[i,j] = 0
                    else:
                        resized_mask[i,j] = 255
            cv2.imwrite(mask_name, resized_mask)

class SpaceNetDataGenerator:
    """Class for generating RGB images from 8-band GeoTIFF images

       Dataset Structure: City_Name
                           |_ geojson/
                           |_ MUL/
                           |_ MUL-PanSharpen
                           |_ PAN
                           |_ RGB-PanSharpen
                           |_ summaryData
    """
    def __init__(self, data_dir, out_dir, patch_size=256, split=True):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.patch_size = patch_size
        self.split = split

    def generate_mask(self, vector_src, raster_scr,
                      outFileName, noDataVal=0, burnVal=1):
        source_ds = ogr.Open(vector_src)
        source_layer = source_ds.GetLayer()

        srcRas_ds = gdal.Open(raster_scr)
        cols = srcRas_ds.RasterXSize
        rows = srcRas_ds.RasterYSize

        memdrv = gdal.GetDriverByName('GTiff')
        dstPath = 'test_gt.tif'

        dst_ds = memdrv.Create(dstPath,
                               cols,
                               rows,
                               gdal.GDT_Byte,
                               options=['COMPRESS=LZW'])

        dst_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
        dst_ds.SetProjection(srcRas_ds.GetProjection())
        band = dst_ds.GetRasterBand(1)
        band.SetNoDataValue(noDataVal)
        gdal.RasterizeLayer(dst_ds, [1], source_layer, burn_values=[1])
        dst_ds = 0

        mask_image = Image.open(dstPath)
        mask_image = np.array(mask_image)
        plt.imsave(outFileName, mask_image, cmap='gray')
        os.remove(dstPath)

    def translate_image(self, raster_src, outFileName):
        translate_options = gdal.TranslateOptions(format='PNG',
                                                  outputType=gdal.GDT_Byte,
                                                  scaleParams=[''],
                                                  )
        gdal.Translate(destName='test.png',
                       srcDS=raster_src,
                       options=translate_options)

        image = Image.open('test.png')
        image = ImageEnhance.Contrast(image)
        image = image.enhance(1.5)
        image = ImageEnhance.Brightness(image)
        image = image.enhance(1.3)
        plt.imsave(outFileName,image)
        os.remove('test.png')

    def spaceNet_data_processing(self, src_root_dir, dst_root_dir):
        vector_path = os.path.join(src_root_dir, 'geojson/')
        raster_path = os.path.join(src_root_dir, 'images/')
        gt_path = dst_root_dir + 'gt/'
        image_path = dst_root_dir + 'images/'

        _, _, vector_files = next(os.walk(vector_path))
        _, _, raster_files = next(os.walk(raster_path))

        if len(vector_files) != len(raster_files):
            print("Check the Directory!!! File(s) missing")
            return -1

        num_of_files = len(vector_files)
        for file in tqdm(raster_files):
            filename, _ = os.path.splitext(file)
            extract_name = filename.split("RGB-PanSharpen")
            raster_filename = "RGB-PanSharpen" + extract_name[1] + '.tif'
            vector_filename = "buildings" + extract_name[1] + '.geojson'

            dst_filename, _ = os.path.splitext(raster_filename)
            dst_filename = dst_filename + '.png'

            gt_filename_path = os.path.join(gt_path, dst_filename)
            image_filename_path = os.path.join(image_path, dst_filename)

            vector_filename = os.path.join(vector_path, vector_filename)
            raster_filename = os.path.join(raster_path, raster_filename)

            self.translate_image(raster_filename, image_filename_path)
            self.generate_mask(vector_filename, raster_filename, gt_filename_path)

    def split(self, image, patch_size):
        image_patches = []
        for i in range(0, image.shape[0], patch_size):
            for j in range(0, image.shape[0], patch_size):
                image_patch = image[j: j+patch_size, i: i+patch_size]
                if image_patch.shape[0] == patch_size and image_patch.shape[1] == patch_size:
                    image_patches.append(image_patch)
        return image_patches

    def save_image(self, image_patches, image_name):
        for img in range(len(image_patches)):
            file_name = image_name + str(img) + '.png'
            cv2.imwrite(file_name, image_patches[img])

    def split_image(self, image, image_name, mask, mask_name, patch_size):
        image_patches = []
        mask_patches = []
        image_patches = self.split(image, patch_size)
        self.save_image(image_patches, image_name)
        mask_patches = self.split(mask, patch_size)
        self.save_image(mask_patches, mask_name)

    def dataset_generator(self):
        image_path = os.path.join(self.data_dir, 'images/')
        mask_path = os.path.join(self.data_dir, 'gt/')

        dst_image_path = os.path.join(self.out_dir, 'images/')
        dst_mask_path = os.path.join(self.out_dir, 'gt/')

        image_patches = []
        mask_patches = []
        _, _, files = next(os.walk(image_path))
        for file in tqdm(files):
            image_name = os.path.join(image_path, file)
            mask_name = os.path.join(mask_path, file)

            image = cv2.imread(image_name)
            mask = cv2.imread(mask_name)

            filename, _ = os.path.splitext(file)

            image_name = dst_image_path + filename + '_'
            mask_name = dst_mask_path + filename + '_'

            self.split_image(image, image_name, mask, mask_name, self.patch_size)

class DataGenerator:
    """Class for generating patches from random dataset.
       The dataset should binary mask and RGb images in the following format
       Dataset Structure: dataset_name
                           |_ images/
                           |_ gt/
    """
    def __init__(self):
        pass
    def generate(self):
        pass
