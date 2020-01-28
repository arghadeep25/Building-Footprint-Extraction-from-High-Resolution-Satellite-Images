"""
Name: Train
Author: Arghadeep Mazumder
Version: 0.1
Description: 1. Train the models
                - FCN
                - SegNet
                - U-Net
                - Deep Residual U-Net
                - PSPNet
             2. Store the training data in a text file
             3. Plot the training metrics
"""
import os
import sys
import numpy as np
sys.path.append('../')
from utils.loader import InriaDataLoader
from models.fcn import FCN
from models.segnet import SegNet
from models.unet import UNet
from models.deepunet import DeepUNet
from models.pspnet import PSPNet
from utils.plot import Plot
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras import backend as keras
from keras.utils import plot_model, Sequence
import keras.backend.tensorflow_backend as K
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class Train:
    def __init__(self, train_path, validation_path, model_name,
                 patch_size, activate_aug=True, rotation=0, sigma = 0,
                 zoom_range=1, horizontal_flip=False, vertical_flip=False,
                 shear=0, brightness=False, add_noise=False,
                 hist_eq=False, epochs=5, batch_size=8,
                 learning_rate=0.001, pre_trained=False):
        self.train_path = train_path
        self.validation_path = validation_path
        self.model_name = model_name
        self.patch_size = patch_size
        self.activate_aug = activate_aug
        self.rotation = rotation
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.shear = shear
        self.brightness = brightness
        self.add_noise = add_noise
        self.hist_eq = hist_eq
        self.sigma = sigma
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pre_trained = pre_trained

    def train(self):
        print('''
Starting Training:

    Train Data Path: {}
    Val Data Path :  {}
    Model:           {}
    Patch Size:      {}
    Augmentation:    {}
    Rotation:        {}
    Zoom Range:      {}
    Horizontal Flip: {}
    Vertical Flip:   {}
    Shear:           {}
    Brightness:      {}
    Noise:           {}
    Hist Eq:         {}
    Sigma:           {}
    Epochs:          {}
    Batch Size:      {}
    Learning Rate:   {}
    Pretrained:      {}

        '''.format(str(self.train_path), str(self.validation_path),
                   str(self.model_name), self.patch_size, self.activate_aug,
                   self.rotation, self.zoom_range, self.horizontal_flip,
                   self.vertical_flip, self.shear, self.brightness,
                   self.add_noise, self.hist_eq, self.sigma, self.epochs,
                   self.batch_size, self.learning_rate, self.pre_trained))

        images_path = os.path.join(self.train_path, 'images/')
        print(self.train_path)
        train_ids = next(os.walk(images_path))[2]

        validation_data_path = os.path.join(self.validation_path, 'images/')
        valid_ids = next(os.walk(validation_data_path))[2]

        if self.model_name == 'fcn':
            models = FCN(pre_trained=self.pre_trained)
            mcp_path_name = 'trained_models/fcn_inria.h5'
            training_data_file = 'training_data/fcn_train_data.txt'
        elif self.model_name == "segnet":
            models = SegNet(pre_trained=self.pre_trained)
            mcp_path_name = 'trained_models/segnet_inria.h5'
            training_data_file = 'training_data/segnet_train_data.txt'
        elif self.model_name == 'unet':
            models = UNet(pre_trained=self.pre_trained)
            mcp_path_name = 'trained_models/unet_inria.h5'
            training_data_file = 'training_data/unet_train_data.txt'
        elif self.model_name == 'deepunet':
            models = DeepUNet(pre_trained=self.pre_trained)
            mcp_path_name = 'trained_models/deepunet_inria.h5'
            training_data_file = 'training_data/deepunet_train_data.txt'
        elif self.model_name == 'pspnet':
            models = PSPNet(pre_trained=self.pre_trained)
            mcp_path_name = 'trained_models/pspnet_inria.h5'
            training_data_file = 'training_data/pspnet_train_data.txt'
        else:
            print('Select a valid model to train...')
            return

        model = models.network()
        if self.pre_trained is False:
            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['acc'])

        model.summary()

        train_gen = InriaDataLoader(data_ids=train_ids,
                                    data_path=self.train_path,
                                    patch_size = self.patch_size,
                                    batch_size = self.batch_size,
                                    aug = self.activate_aug,
                                    rotation = self.rotation,
                                    zoom_range = self.zoom_range,
                                    horizontal_flip = self.horizontal_flip,
                                    vertical_flip = self.vertical_flip,
                                    shear = self.shear,
                                    add_noise = self.add_noise,
                                    brightness = self.brightness,
                                    hist_eq = self.hist_eq,
                                    sigma=self.sigma)

        valid_gen = InriaDataLoader(data_ids=valid_ids,
                                    data_path=self.validation_path,
                                    patch_size=self.patch_size,
                                    batch_size=self.batch_size)

        train_steps = len(train_ids) // self.batch_size
        valid_steps = len(valid_ids) // self.batch_size

        earlyStopping = EarlyStopping(monitor='val_loss',
                                      patience=5,
                                      verbose=0,
                                      mode='min')
        mcp_save = ModelCheckpoint(mcp_path_name,
                                   save_best_only=True,
                                   monitor='val_loss',
                                   mode='min')
        H = model.fit_generator(train_gen,
                                validation_data=valid_gen,
                                steps_per_epoch=train_steps,
                                validation_steps=valid_steps,
                                callbacks=[earlyStopping, mcp_save],
                                epochs=self.epochs)

        training_loss_array = np.asarray(H.history['loss'])
        validation_loss_array = np.asarray(H.history['val_loss'])
        training_accuracy_array = np.asarray(H.history['acc'])
        validation_accuracy_array = np.asarray(H.history['val_acc'])
        epoch = []
        for i in range(len(training_loss_array)):
            epoch.append(i+1)

        if self.pre_trained is True:
            if os.path.exists(training_data_file):
                test_file = open(training_data_file, "r")
                lines = test_file.readlines()
                new_epochs= []
                new_training_loss = []
                new_validation_loss = []
                new_training_accuracy = []
                new_validation_accuracy = []

                for line in lines:
                    new_epochs.append(int(line.split(' ')[0]))
                    new_training_loss.append(float(line.split(' ')[1]))
                    new_validation_loss.append(float(line.split(' ')[2]))
                    new_training_accuracy.append(float(line.split(' ')[3]))
                    new_validation_accuracy.append(float(line.split(' ')[4]))
                test_file.close()

                epochs_last_val = new_epochs[len(new_epochs) -1]
                for i in range(len(training_loss_array)):
                    new_epochs.append(i + epochs_last_val + 1)
                    new_training_loss.append(training_loss_array[i])
                    new_validation_loss.append(validation_loss_array[i])
                    new_training_accuracy.append(training_accuracy_array[i])
                    new_validation_accuracy.append(validation_accuracy_array[i])

                    data_mod = np.array([new_epochs,
                                         new_training_loss,
                                         new_validation_loss,
                                         new_training_accuracy,
                                         new_validation_accuracy])

                data_mod = data_mod.T

                with open(training_data_file, 'w+') as file:
                    np.savetxt(file,
                               data_mod,
                               fmt = ['%d', '%f', '%f', '%f', '%f'])
            else:
                data = np.array([epoch,
                                 training_loss_array,
                                 validation_loss_array,
                                 training_accuracy_array,
                                 validation_accuracy_array])
                data = data.T
                with open(training_data_file, 'w+') as file:
                    np.savetxt(file,
                               data,
                               fmt=['%d', '%f', '%f', '%f', '%f'])
        else:
            data = np.array([epoch,
                             training_loss_array,
                             validation_loss_array,
                             training_accuracy_array,
                             validation_accuracy_array])
            data = data.T
            with open(training_data_file, 'w+') as file:
                np.savetxt(file,
                           data,
                           fmt=['%d', '%f', '%f', '%f', '%f'])
        plot_data = Plot(model=self.model_name)
        plot_data.plot()
