import numpy as np
import cv2
import sys
import os

path = sys.argv[1]
# _, _, files = next(os.walk(path))
image = cv2.imread(path)
image_patches = []

# image_patch = np.zeros((256, 256, 3))
patch_size = 768
image_patche = np.zeros((patch_size, patch_size, 3))
# for file in files:
#     filename = os.path.join(path, file)
#     image = cv2.imread(filename)
#     name = os.path.splitext(file)[0]
#     for i in range(0, image.shape[0], 256):
#         for j in range(0, image.shape[1], 256):
#             image_patch = image[j : j + 256, i : i + 256]
#             # if image_patch.shape[0] == 256 and image_patch.shape[1] == 256:
#             image_patches.append(image_patch)
#             image_name = '../prediction/isprs/gt/' + name + str(i) + '.png'
#             cv2.imwrite(image_name,image_patch)
for i in range(0, image.shape[0], patch_size):
    for j in range(0, image.shape[0], patch_size):
        image_patch = image[j: j+patch_size, i: i+patch_size]
        if image_patch.shape[0] == patch_size and image_patch.shape[1] == patch_size:
            image_patches.append(image_patch)
print(len(image_patches))
i=0
for patch in image_patches:
    i = i+1
    # name = '../prediction/test/images/vaihingen_' + str(i) + '.png'
    name = '../prediction/test/gt/vaihingen_' + str(i) + '.png'
    cv2.imwrite(name, patch)
