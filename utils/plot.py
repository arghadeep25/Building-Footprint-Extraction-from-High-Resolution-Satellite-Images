"""
Name: Plot
Author: Arghadeep Mazumder
Version: 0.1
Description: 1. Plot the train models metrics
                - FCN
                - SegNet
                - U-Net
                - Deep Residual U-Net
                - PSPNet
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

class Plot:
    """Class for plotting the training metrics

       Note: This function is helpful when the training
       data is large and takes long time for one epoch.
    """
    def __init__(self, model):
        self.model = model

    def plot(self):
        data_path = 'training_summary/'
        dstpath = 'training_plots/'
        if self.model == 'fcn':
            file_name = data_path + 'fcn_train_data.txt'
        elif self.model == 'segnet':
            file_name = data_path + 'segnet_train_data.txt'
        elif self.model == 'unet':
            file_name = data_path + 'unet_train_data.txt'
            dstName = 'unet.png'
            dstName = os.path.join(dstpath, dstName)
        elif self.model == 'deepunet':
            file_name = data_path + 'deepunet_train_data.txt'
        elif self.model == 'pspnet':
            file_name = data_path + 'pspnet_train_data.txt'
        else:
            print('Check data path...')

        # dirName = os.path.dirname(file_name)
        # name = os.path.basename(file_name)
        # name, _ = os.path.splitext(name)
        # name = name.split("_data")[0]
        # dstName = name + '.png'
        # dstName = os.path.join(dirName, dstName)

        if os.path.exists(file_name):
            test_file = open(file_name, "r")
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

        fig = plt.figure(figsize = (8, 4))
        ax = plt.subplot(111)

        N = len(new_epochs)

        ax.plot(np.arange(0, N),
                new_training_loss,
                label="Training Loss")
        ax.plot(np.arange(0, N),
                new_validation_loss,
                label="Validation Loss")
        ax.plot(np.arange(0, N),
                new_training_accuracy,
                label="Training Accuracy")
        ax.plot(np.arange(0, N),
                new_validation_accuracy,
                label="Validation Accuracy")

        plt.xlim((0, int(N)))
        plt.ylim((0, 1.0))

        plt.title("Training Loss and Accuracy", fontsize=15)
        plt.xlabel("Epochs", fontsize = 13)
        plt.ylabel("Loss/Accuracy", fontsize = 13)

        ax.legend(loc='upper center',
                  bbox_to_anchor=(0.5, -0.15),
                  frameon=False,
                  ncol=4,
                  prop={'size': 10})

        plt.savefig(dstName, bbox_inches='tight', dpi=100)
        plt.show()
