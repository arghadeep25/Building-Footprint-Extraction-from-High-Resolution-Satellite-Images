import numpy as np
import cv2
import sys
import os

path = sys.argv[1]
save_dir = sys.argv[2]
_, _, files = next(os.walk(path))

for file in files:
    name = os.path.splitext(file)[0]
    mask_name = os.path.join(path, file)
    mask = cv2.imread(mask_name)
    print(mask_name)
    building_mask = np.zeros((mask.shape[0], mask.shape[1]))
    save_mask_name = save_dir + name + '.png'
    print(save_mask_name)
# for ch in range(mask.shape[2]):
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j,0] == 255 and mask[i,j,1] == 0 and mask[i,j,2] == 0:
                building_mask[i,j] = 255
    cv2.imwrite(save_mask_name,building_mask)
# mask[]
