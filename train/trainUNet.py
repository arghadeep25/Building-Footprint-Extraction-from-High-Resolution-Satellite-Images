"""
Name: Train U-Net Model
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
import time
import torch
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
sys.path.append('../')
from utils.loader import InriaDataLoader
from models.unet import UNet

class TrainUNet():
    """ Class for training U-Net Model

        Parameters: data_path       = train folder
                    patch_size      = 256x256
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
    def  __init__(self, data_path, net, patch_size = 256,
                activate_aug = False, rotation = 0,
                zoom_range = 1, horizontal_flip = False,
                vertical_flip = False, shear = 0,
                epochs = 5, batch_size = 1,
                learning_rate = 0.1, val_percent = 0.05,
                save_model = False, activate_gpu = False):

        self.data_path = data_path
        self.net = net
        self.patch_size = patch_size
        self.activate_aug = activate_aug
        self.rotation = rotation
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.shear = shear
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_percent = val_percent
        self.save_model = save_model
        self.activate_gpu = activate_gpu

    def train(self):
        data = InriaDataLoader(self.data_path,
                            patch_size = self.patch_size,
                            aug = self.activate_aug,
                            rotation = self.rotation,
                            zoom_range = self.zoom_range,
                            horizontal_flip = self.horizontal_flip,
                            vertical_flip = self.vertical_flip,
                            shear = self.shear)

        images, masks = data.__getitem__()
        total_data = len(images)
        optimizer = optim.SGD(self.net.parameters(),
                            lr = self.learning_rate,
                            momentum=0.9,
                            weight_decay=0.0005)

        criterion = nn.BCELoss()

        for epoch in range(self.epochs):
            self.net.train()
            epoch_loss = 0
            print('Training')
            for i in range(total_data):
                image = images[i]
                image = torch.from_numpy(image)
                mask = masks[i]
                mask = torch.from_numpy(mask)

                if self.activate_gpu:
                    image = image.cuda()
                    mask = mask.cuda()

                mask_pred = self.net(image)
                prob_mask_flat = mask_pred.view(-1)
                true_mask_flat = mask.view(-1)

                loss = criterion(prob_mask_flat, true_mask_flat)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if self.save_model:
            torch.save(net.state_dict(),path + 'CP{}.pth'.format(epoch + 1))
        print('Training Successful')
