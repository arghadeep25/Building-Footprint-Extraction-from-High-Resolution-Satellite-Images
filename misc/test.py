# import numpy as np
# import cv2
# import sys
# import matplotlib.pyplot as plt
#
# image = cv2.imread(sys.argv[1])
#
# patches = []
#
# image_patches = np.zeros((256, 256, 3))
#
# for img_col in range(0, image.shape[0], 256):
#     for img_row in range(0, image.shape[1], 256):
#         image_patch = image[img_row:img_row+256, img_col:img_col+256]
#         if image_patch.shape[0] == 256 and image_patch.shape[1] == 256:
#             patches.append(image_patch)
#
# for i in range(len(patches)):
#     name = 'Test_' + str(i) + '.png'
#     cv2.imwrite(name, patches[i])

import gdal
import sys
import numpy as np
import cv2
import png
import matplotlib.pyplot as plt

geo = gdal.Open(sys.argv[1])
min = []
max = []

image = np.zeros((40155, 40180, 3))
for i in range(3):
    image[:,:,i] = geo.GetRasterBand(i+1).ReadAsArray()

for i in range(3):
    val_min = np.min(image[:,:,i])
    val_max = np.max(image[:,:,i])
    min.append(val_min)
    max.append(val_max)

for i in range(3):
    image[:,:,i] = image[:,:,i] * (255.0/max[i])

# image = np.uint8(image)
# image = cv2.clip
# image = image.astype(np.uint8)
# image = image/255
# image = np.uint8(image)
# image = image.astype(np.uint16)
# image = np.transpose(image)


# image[:,:,0] = cv2.normalize(image[:,:,0], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX)
# image[:,:,1] = cv2.normalize(image[:,:,1], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX)
# image[:,:,2] = cv2.normalize(image[:,:,2], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX)
min_norm = []
for i in range(3):
    min_val = np.min(image[:,:,i])
    min_norm.append(min_val)
    print(min_val)

for ch in range(3):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i,j,ch] = ((image[i,j,ch] - min_norm[ch])*255.0) / (255.0 - min_norm[ch]+1)
# # # image = cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
image = image.astype(np.uint8)
# image[:,:,0] = cv2.equalizeHist(image[:,:,0])
# image[:,:,1] = cv2.equalizeHist(image[:,:,1])
# image[:,:,2] = cv2.equalizeHist(image[:,:,2])


color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([image],[i], None, [256], [0,256])
    plt.plot(histr,color = col)
    plt.xlim([0, 256])
plt.show()
#
cv2.imshow('Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# # print(image.shape)
# print(np.max(image[:,:,1]))
# # print((np.transpose(image)).shape)

# with open('test.png', 'wb') as f:
#     writer = png.Writer(width=650, height=650, bitdepth=16)
#     z2list = image.reshape(-1, 650*650).tolist()
#     writer.write(f, z2list)
