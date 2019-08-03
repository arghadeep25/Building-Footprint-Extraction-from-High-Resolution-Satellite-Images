"""
Name: Boundaries
Author: Arghadeep Mazumder
Version: 0.1
Description:
    PolyApprox: for approximating boundaries
    BoudingBox: for approximating bounding box
"""
import cv2
import numpy as np


class PolyApprox():
    """
    Class for approximating boundaries of polygons in an image

    Parameters: Image
                Mask
    """
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask

    def approx(self):
        _, _, channel = self.mask.shape

        if channel == 3 or None:
            self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
            self.mask = np.uint8(self.mask)
            self.mask = np.expand_dims(self.mask, 3)

        contours, _ = cv2.findContours(self.mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            eps = 0.01 * cv2.arcLength(contour, True)
            shapes = cv2.approxPolyDp(contour, eps, True)
            output = cv2.drawContours(self.image,
                                      [shapes],
                                      0, (0, 0, 255), 2)

        return output


class BoudingBox():
    """
    Class for approximating bounding boxes of polygons in an image

    Parameters: Image
                Mask
    """
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask

    def approx(self):
        _, _, channel = self.mask.shape

        if channel == 3 or None:
            self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
            self.mask = np.uint8(self.mask)
            self.mask = np.expand_dims(self.mask, 3)

        contours, _ = cv2.findContours(self.mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            output = cv2.drawContours(self.image,
                                      [box],
                                      0, (0, 0, 255), 2)
        return output
