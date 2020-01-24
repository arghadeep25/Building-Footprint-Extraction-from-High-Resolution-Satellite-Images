"""
Name: Intersection over Union (IoU)
Author: Arghadeep Mazumder
Version: 0.1
Description:
"""
import numpy as np


class IoU:
    def __init__(self, gt, predicted):
        self.gt = gt
        self.predicted = predicted

    def calculate(self):
        if self.gt.shape[0] == self.predicted.shape[0] and \
           self.gt.shape[1] == self.predicted.shape[1]:
            intersection = np.logical_and(self.gt, self.predicted)
            union = np.logical_or(self.gt, self.predicted)
            iou_score = np.sum(intersection) / np.sum(union)
            return iou_score
        else:
            print("Check the size of GT and Prediction...")
            return
