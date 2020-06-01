"""
Name: Pixel Accuracy
Author: Arghadeep Mazumder
Version: 0.1
Description:
"""
import numpy as np

class Accuracy:
    """Class for computing pixel-wise accuracy
    """
    def __init__(self, gt: np.ndarray, predicted: np.ndarray) -> None:
        self.gt = gt
        self.predicted = predicted

    def extract_masks(self, mask: np.ndarray,
                      cl: int, patch_size: np.ndarray) -> np.ndarray:
        new_mask = np.zeros((len(cl),patch_size, patch_size))
        for i, c in enumerate(cl):
            new_mask[i,:,:] = mask == c
        return new_mask

    def calculate(self) -> float:
        patch_size = self.gt.shape[0] #considering square patches
        if self.gt.shape[0] == self.predicted.shape[0] and \
        self.gt.shape[1] == self.predicted.shape[1]:

            cl = np.unique(self.gt)
            n_cl = len(cl)
            pred_mask = self.extract_masks(self.predicted, cl, patch_size)
            gt_mask = self.extract_masks(self.gt, cl, patch_size)

            sum_n_ii = 0
            sum_t_i = 0

            for i, c in enumerate(cl):
                curr_pred_mask = pred_mask[i,:,:]
                curr_gt_mask = gt_mask[i,:,:]

                sum_n_ii += np.sum(np.logical_and(curr_pred_mask, curr_gt_mask))
                sum_t_i += np.sum(curr_gt_mask)
            if sum_t_i == 0:
                pixel_accuracy_val = 0
            else:
                pixel_accuracy_val = sum_n_ii/sum_t_i
            return pixel_accuracy_val
        else:
            print("Check the size of GT and Prediction...")
            return
