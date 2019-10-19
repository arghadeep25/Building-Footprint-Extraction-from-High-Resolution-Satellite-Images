import numpy as np
import cv2
import os
import sys

image = cv2.imread(sys.argv[1])

# image[:,:,0] = cv2.equalizeHist(image[:,:,0])
# image[:,:,1] = cv2.equalizeHist(image[:,:,1])
# image[:,:,2] = cv2.equalizeHist(image[:,:,2])
#
# max_b = np.max(image[:,:,0])
# max_g = np.max(image[:,:,1])
# max_r = np.max(image[:,:,2])
#
# min_b = np.min(image[:,:,0])
# min_g = np.min(image[:,:,1])
# min_r = np.min(image[:,:,2])
#
# print("Max R: ", max_r)
# print("Max G: ", max_g)
# print("Max B: ", max_b)
#
# print("Min R: ", min_r)
# print("Min G: ", min_g)
# print("Min B: ", min_b)
#
# kernel_sharp = np.array([[-1, -1, -1],
#                          [-1, 9, -1],
#                          [-1, -1, -1]])
# image = cv2.filter2D(image, -1, kernel_sharp)

# Change the brightnes of the image
# image[:,:,0] = image[:,:,0] * 0.5
# image[:,:,1] = image[:,:,1] * 0.5
# image[:,:,2] = image[:,:,2] * 0.5

# Add Noise
row, col, ch = image.shape
mean = 0
var = 0.1
sigma = var**0.5
gauss = np.random.normal(mean,sigma,(row, col, ch))
gauss = gauss.reshape(row, col, ch)*75
print('Gauss: ', gauss[10:15, 10:15, 1])
image[:,:,0] = image[:,:,0] + gauss[:,:,0]
image[:,:,1] = image[:,:,1] + gauss[:,:,1]
image[:,:,2] = image[:,:,2] + gauss[:,:,2]
# image[:,:,0] = image[:,:,0] - 127.5
# image[:,:,1] = image[:,:,1] - 127.5
# image[:,:,2] = image[:,:,2] - 127.5
cv2.imwrite('austin1_mod.tif',image)
