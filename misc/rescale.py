import numpy as np
import sys
import cv2
from skimage.transform import resize, rescale
from scipy.ndimage import zoom
import time

def zoom_image(image, zoom_factor):
    if len(image.shape) == 3:
        height, width, channel = image.shape
    else:
        height, width = image.shape
    print(len(image.shape))
    zoom_factor = 1.5
    zoom_tuple = (zoom_factor,)*2 + (1,)*(image.ndim - 2)
    print(zoom_tuple)
    z_height = int(np.round(height / zoom_factor))
    z_width = int(np.round(height / zoom_factor))
    top = (height - z_height) // 2
    left = (width - z_width) // 2

    out = zoom(image[top:top+z_height, left:left+z_width], zoom_tuple)

    trim_top = ((out.shape[0] - height) // 2)
    trim_left = ((out.shape[1] - width) // 2)
    out = out[trim_top:trim_top+height, trim_left:trim_left+width]

    return out

image = cv2.imread(sys.argv[1])
mask = cv2.imread(sys.argv[2])
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# resized_mask = rescale(image, 1.5)
resized_image = zoom_image(image, 1.5)
resized_mask = zoom_image(mask, 1.5)
start = time.time()
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if resized_mask[i,j] < 200:
            resized_mask[i,j] = 0
        else:
            resized_mask[i,j] = 255
end = time.time()
print('Time Taken: ',(end-start))
cv2.imshow('image', image)
cv2.imshow('Resized', resized_image)
cv2.imshow('Mask',mask)
cv2.imshow('Resized Mask',resized_mask)
print(np.unique(resized_mask))
cv2.waitKey(0)
cv2.destroyAllWindows()
