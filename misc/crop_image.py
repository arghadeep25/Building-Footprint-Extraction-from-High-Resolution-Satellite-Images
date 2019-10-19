import numpy as np
import cv2
import sys


img = cv2.imread(sys.argv[1])
cropped_img = img[0:99, :]
cv2.imshow('Original', img)
cv2.imshow('Cropped',cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()