"""
Name: Train U-Net Model
Author: Arghadeep Mazumder
Version: 0.1
Description: Train U-Net Model

"""
import os
import sys
import cv2
import keras.utils
import numpy as np
sys.path.append('../')
from utils.iou import IoU
from models.unet import UNet
from utils.loader import InriaDataLoader
from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from utils.visualizer import InriaVisualizer

class TrainUNet():
    """ Class for training U-Net Model

        Parameters: train_path       = train folder
                    patch_size      = 256
                    activate_aug    = False
                    rotation        = 0
                    zoom_range      = 1
                    horizontal_flip = False
                    vertical_flip   = False
                    shear           = 0
                    net             = U-Net
                    epochs          = 5
                    batch_size      = 1
                    learning_rate   = 0.2
                    val_percent     =
                    save_model      = False
                    activate_gpu    = False
    """
    def __init__(self,
                train_path,
                image_size = 256,
                activate_aug = False,
                rotation = 0,
                zoom_range = 1,
                horizontal_flip = False,
                vertical_flip = False,
                shear = 0,
                epochs = 10,
                batch_size = 8,
                learning_rate = 0.1,
                save_model = True,
                val_data_size = 10,
                evaluate = True):

        self.train_path = train_path
        self.image_size = image_size
        self.activate_aug = activate_aug
        self.rotation = rotation
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.shear = shear
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_model = save_model
        self.val_data_size = val_data_size
        self.evaluate = evaluate

    def train(self):
        print('''
Start Training:

    Data Path:       {}
    Model:           {}
    Patch Size:      {}
    Augmentation:    {}
    Rotation:        {}
    Zoom Range:      {}
    Horizontal Flip: {}
    Vertical Flip:   {}
    Shear:           {}
    Epochs:          {}
    Batch Size:      {}
    Learning Rate:   {}
    Validation %:    {}
    Save Model:      {}
    Evaluate:        {}

        '''.format(str(self.train_path), str('U-Net'), self.image_size,
            self.activate_aug, self.rotation, self.zoom_range,
            self.horizontal_flip, self.vertical_flip, self.shear,
            self.epochs, self.batch_size, self.learning_rate,
            self.val_data_size, self.save_model, self.evaluate))

        images_path = os.path.join(self.train_path, 'images/')
        train_ids = next(os.walk(images_path))[2]

        valid_ids = train_ids[:self.val_data_size]
        train_ids = train_ids[self.val_data_size:]


        models = UNet()
        model = models.network()
        model.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['acc'])
        model.summary()

        train_gen = InriaDataLoader(train_ids,
                                    self.train_path,
                                    patch_size = self.image_size,
                                    batch_size = self.batch_size)

        valid_gen = InriaDataLoader(valid_ids,
                                    self.train_path,
                                    patch_size = self.image_size,
                                    batch_size = self.batch_size)

        train_steps = len(train_ids) // self.batch_size
        valid_steps = len(valid_ids) // self.batch_size
        print(train_steps)
        model.fit_generator(train_gen,
                            validation_data = valid_gen,
                            steps_per_epoch = train_steps,
                            validation_steps = valid_steps,
                            epochs = self.epochs)
        if self.save_model:
            model.save('building_unet.h5')

        keras.utils.plot_model(model, to_file = 'unet_architecture.png')

        if self.evaluate:
            image , mask = valid_gen.__getitem__(2)
            result = model.predict(x)
            result = result > 0.5
            viz = InriaVisualizer(mask, result)
            viz.plot()

        iou_score = IoU(target= mask, prediction=result)
        print('IoU Score: ',iou_score)
