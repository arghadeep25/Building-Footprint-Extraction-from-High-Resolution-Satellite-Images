import numpy as np
import cv2
import sys
import os

path = sys.argv[1]
save_dir = sys.argv[2]
image_path = path + 'images/'
mask_path = path + 'gt/'

_, _, files = next(os.walk(image_path))

for file in files:
    image_name = os.path.join(image_path, file)
    image = cv2.imread(image_name)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
    save_image_name = save_dir + 'images/' + file
    cv2.imwrite(save_image_name, image)

    mask_name = os.path.join(mask_path, file)
    mask = cv2.imread(mask_name)
    # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    save_mask_name = save_dir + 'gt/' + file
    cv2.imwrite(save_mask_name, mask)
