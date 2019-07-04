import numpy as np


class IoU():
    """ IoU class computes the IoU score between the ground truth
        and the predicted mask

        Parameters: target
                    prediciton
    """
    def __init__(self, target, prediction):
        self.target = target
        self.prediction = prediction

    def compute(self):
        intersection = np.logical_and(self.target, self.prediction)
        union = np.logical_or(self.target, self.prediction)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score
