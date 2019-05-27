import os
import sys
import time
import cv2
from utils import loader
from models.unet import UNet
from train.trainUNet import TrainUNet

def main():
    # path = sys.argv[1]
    #
    # data = loader.InriaDataLoader(path, rotation =90, aug = True)
    # images, masks = data.__getitem__()
    #
    # i =30
    # cv2.imshow('Image', images[i])
    # cv2.imshow('Mask', masks[i])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(len(images))
    path = sys.argv[1]

    net = UNet(n_channels = 3, n_classes = 1)
    train_unet = TrainUNet(data_path= path, net = net)
    train_unet.train()

if __name__ == '__main__':
    main()
