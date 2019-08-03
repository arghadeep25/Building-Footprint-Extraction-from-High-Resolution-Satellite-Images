"""
Name: Evaluate U-Net Model
Author: Arghadeep Mazumder
Version: 0.1
Description: Evaluation of U-Net trained weights

"""
import os
import cv2
import sys
sys.path.append('../')
import numpy as np
from keras.models import load_model
from utils.loader import InriaDataLoader
import matplotlib.pyplot as plt


class EvalUNet():
    """
    Class for evaluating the trained models of U-Net

    Parameters: data_path (test images path)
                weight_path (path of trained models)
                weight_name (Optional)
    """
    def __init__(self, data_path, weight_path, weight_name = 'building_unet.h5'):
        self.data_path = data_path
        self.weight_path = weight_path
        self.weight_name = weight_name

    def evaluate(self):
        weight = os.path.join(self.weight_path, self.weight_name)
        print(weight)
        model = load_model(weight)
        model.summary()
        model.get_weights()

        images_path = os.path.join(self.data_path, 'images/')
        train_ids = next(os.walk(images_path))[2]
        valid_ids = train_ids[:10]
        print(valid_ids)
        val_data = InriaDataLoader(valid_ids,
                                    self.data_path,
                                    patch_size = 256,
                                    batch_size = 8)
        x, y = val_data.__getitem__(0)
        print(x.shape)
        # cv2.imshow('img',x)
        result = model.predict(x)
        result = result > 0.5
        plt.imshow(np.reshape(result[0]*255,(256, 256)))
        plt.show()
        print(np.reshape(result[0]*255,(256, 256)))

def main():
    data = sys.argv[1]
    path = sys.argv[2]
    eval = EvalUNet(data_path = data ,weight_path=path)
    eval.evaluate()

if __name__ == '__main__':
    main()
