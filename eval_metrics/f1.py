"""
Name: Dice Score or F1 Score
Author: Arghadeep Mazumder
Version: 0.1
Description:
"""
import numpy as np


class F1:
    def __init__(self, gt, predicted):
        self.gt = gt
        self.predicted = predicted

    def calculate(self):
        if self.gt.shape[0] == self.predicted.shape[0] and \
           self.gt.shape[1] == self.predicted.shape[1]:
            dice_score = np.sum(self.predicted[self.gt == 255])*2.0 / \
            (np.sum(self.predicted) + np.sum(self.gt))
            return dice_score
        else:
            print("Check the size of GT and Prediction...")
            return
