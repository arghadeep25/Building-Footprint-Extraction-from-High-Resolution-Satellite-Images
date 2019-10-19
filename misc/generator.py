import numpy as np
import cv2
import sys
import random


random.seed(0)

path = sys.argv[1]

img = cv2.imread(path)
height, width, channel = img.shape

nclasses = len(np.unique(img))
class_color = [(random.randint(0,255),
                random.randint(0,255),
                random.randint(0,255))
                for _ in range(5)]

mask = np.zeros((height, width, 2))
img = (img[:, :, 1])/255
for c in range(nclasses):
    mask[:, :, c] = (img == c).astype(int)
print(np.unique(mask))
cv2.imshow('Mask',mask[:,:,1])
cv2.waitKey(0)
cv2.destroyAllWindows()
# mask = cv2.resize(mask, (100, 100), interpolation = cv2.INTER_NEAREST)
#
# # mask = np.reshape(mask, (100*100, 2))
#
# print(mask.shape)
#
# result_image = np.zeros((100, 100, 3))
#
# result_image[:, :, 0] = mask[:, :, 0]
# result_image[: ,:, 1] = mask[:, :, 1]
#
# result_image = cv2.resize(result_image, (256, 256), interpolation=cv2.INTER_NEAREST)
# cv2.imshow('result', result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(np.unique(img))
# print('Shape of mask',mask.shape)
# for i in range(nclasses):
#     print(np.unique(mask[:,:,i]))
#     cv2.imshow('Mask',mask[:,:,i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# cv2.imshow('Another',mask[:,:,1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# test_data = np.zeros((height, width, 3))
# for c in range(nclasses):
#     test_data[:,:,0] += ((mask[:,:,0] == c)*(class_color[c][0])).astype('uint8')
#     test_data[:,:,1] += ((mask[:,:,0] == c)*(class_color[c][1])).astype('uint8')
#     test_data[:,:,2] += ((mask[:,:,0] == c)*(class_color[c][2])).astype('uint8')

# print(mask.shape)
# cv2.imshow('test',test_data/255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
