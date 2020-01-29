"""
Name: Predict
Author: Arghadeep Mazumder
Version: 0.1
Description: 1. Predict from the following trained models
                - FCN
                - SegNet
                - U-Net
                - Deep Residual U-Net
                - PSPNet
             2. Overlay mask on the input images
             3. Estimate the evaluation metrics
"""
import os
import cv2
import sys
import time
import numpy as np
sys.path.append('../')
from eval_metrics.f1 import F1
from eval_metrics.iou import IoU
from eval_metrics.accuracy import Accuracy
from keras.models import Model, load_model
from keras.preprocessing.image import img_to_array

class Predict:
    """Class for predicting results from trained models
    """
    def __init__(self, model_name, data_path, output_path):
        self.model_name = model_name
        self.data_path = data_path
        self.output_path = output_path

    def eval(self):
        model_dir = 'trained_models/'
        if self.model_name == 'fcn':
            model_path = model_dir + 'fcn_inria.h5'
            model = load_model(model_path)
        elif self.model_name == 'segnet':
            model_path = model_dir + 'segnet_inria.h5'
            model = load_model(model_path)
        elif self.model_name == 'unet':
            model_path = model_dir + 'unet_inria.h5'
            model = load_model(model_path)
        elif self.model_name == 'deepunet':
            model_path = model_dir + 'deepunet_inria.h5'
            model = load_model(model_path)
        elif self.model_name == 'pspnet':
            model_path = model_dir + 'pspnet_inria.h5'
            model = load_model(model_path)
        else:
            print('Check model name...')

        image_path = os.path.join(self.data_path + 'images/')
        mask_path = os.path.join(self.data_path, 'gt/')

        _, _, files = next(os.walk(image_path))

        if self.model_name == 'pspnet':
            result = np.zeros((384, 384, 3))
        else:
            result = np.zeros((256, 256, 3))

        avg_iou = []
        avg_dice = []
        avg_pixel_acc = []
        time_list = []

        for file in files:
            name = os.path.splitext(file)[0]
            image_name = image_path + file
            mask_name = mask_path + file
            image = cv2.imread(image_name)
            mask = cv2.imread(mask_name)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            original_image = image.astype(np.uint8)
            if self.model_name == 'pspnet':
                original_image = cv2.resize(original_image, (384, 384))
                image = cv2.resize(image, (384, 384))
                mask = cv2.resize(mask, (384, 384))
            else:
                original_image = cv2.resize(original_image, (256, 256))
                image = cv2.resize(image, (256, 256))
                mask = cv2.resize(mask, (256, 256))

            image = img_to_array(image)/255.
            image = np.expand_dims(image, 0)
            start = time.time()
            prediction = model.predict(image)
            end = time.time()
            time_com = end - start
            prediction = prediction > 0.5

            if self.model_name == 'segnet':
                prediction = prediction*255
                prediction = prediction.reshape((256, 256, 2))
                prediction = prediction[:,:,1]
            elif self.model_name == 'pspnet':
                prediction = np.reshape(prediction[0]*255, (384, 384, 2))
                prediction = prediction[:,:,1]
            else:
                prediction = np.reshape(prediction[0]*255, (256, 256))

            prediction = prediction.astype(np.uint8)
            iou = IoU(prediction, mask)
            iou_score = iou.calculate()

            dice = F1(prediction, mask)
            dice_score = dice.calculate()

            accuracy = Accuracy(prediction, mask)
            pixel_accuracy_score = accuracy.calculate()

            time_list.append(time_com)

            if iou_score > 0.2:
                avg_iou.append(iou_score)
            if dice_score > 0.2:
                avg_dice.append(dice_score)
            if pixel_accuracy_score > 0.2:
                avg_pixel_acc.append(pixel_accuracy_score)

            result[:, :, 2] = prediction
            result = result.astype(np.uint8)

            overlay = cv2.addWeighted(result, 1, original_image, 1, 0)
            cv2.imwrite(self.output_path + name +'.png', overlay)

        min_iou = np.min(avg_iou)
        print('Min IoU: ',min_iou)
        avg_score = sum(avg_iou)/len(avg_iou)
        print('Average IoU: ',avg_score)
        max_iou = np.max(avg_iou)
        print('Max IoU: ',max_iou)

        print('\n')

        min_dice_score = np.min(avg_dice)
        print('Min Dice: ',min_dice_score)
        avg_dice_score = sum(avg_dice)/len(avg_dice)
        print('Average Dice Score: ',avg_dice_score)
        max_dice_score = np.max(avg_dice)
        print('Max Dice: ',max_dice_score)

        print('\n')

        min_acc = np.min(avg_pixel_acc)
        print('Min Acc: ',min_acc)
        avg_pixel_score = sum(avg_pixel_acc)/len(avg_pixel_acc)
        print('Average Pixel Accuracy: ',avg_pixel_score)
        max_acc = np.max(avg_pixel_acc)
        print('Max Acc: ', max_acc)

        print('\n')

        time_avg = sum(time_list)/len(time_list)
        print('Prediction Time: ',time_avg)
