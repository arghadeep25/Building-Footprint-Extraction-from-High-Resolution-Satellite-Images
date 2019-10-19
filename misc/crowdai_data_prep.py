import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import skimage.io as io
from skimage.transform import resize
import matplotlib.pyplot as plt
import pylab
import random
import os
import sys

data_directory = '/media/arghadeep25/Arghadeep/TU Berlin/8 Thesis/Datasets/CrowdAI Dataset Building/'
annotation_file_template = '{}/{}/annotation{}.json'

# Train images and validation path
train_images_dir = data_directory + 'train/images'
train_annotation_path = data_directory + 'train/annotation.json'
train_annotation_small_path = data_directory + 'train/annotation-small.json'

# Validation images and annotation path
val_images_dir = data_directory + 'val/images'
val_annotation_path = data_directory + 'val/annotation.json'
val_annotation_small_path = data_directory + 'val/annotation-small.json'

# Prepared dataset directory
prep_data_directory = '/home/arghadeep25/python_exp/prediction/crowdai/'

# Loading annotation path into memory
coco = COCO(train_annotation_small_path)

category_ids = coco.loadCats(coco.getCatIds())

# collecting and visualizing images
# Generating lists of all images
image_ids = coco.getImgIds(catIds=coco.getCatIds())
# Randomly picking an image
random_image_id = random.choice(image_ids)
print('Random Image Id',random_image_id)


for image_id in image_ids:
    img = coco.loadImgs(image_id)[0]
    image_name = prep_data_directory + 'images/' + str(image_id) + '.png'
    mask_name = prep_data_directory + 'gt/' + str(image_id) + '.png'
    image_path = os.path.join(train_images_dir, img['file_name'])
    I = cv2.imread(image_path)
    annotation_ids = coco.getAnnIds(imgIds=img['id'])
    annotations = coco.loadAnns(annotation_ids)
    mask = np.zeros((300, 300))
    for _idx, annotation in enumerate(annotations):
        rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
        m = cocomask.decode(rle)
        m = m.reshape((img['height'], img['width']))
        mask = np.maximum(mask, m)
    plt.imsave(mask_name, mask)
    resized_img = cv2.resize(I, (256, 256), interpolation=cv2.INTER_NEAREST)
    # resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_name, resized_img)

    resized_mask = cv2.imread(mask_name)
    resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2GRAY)
    resized_mask = cv2.resize(resized_mask, (256, 256), cv2.INTER_NEAREST)
    for i in range(resized_mask.shape[0]):
        for j in range(resized_mask.shape[1]):
            if resized_mask[i,j] <= 70:
                resized_mask[i,j] = 0
            else:
                resized_mask[i,j] = 255
    cv2.imwrite(mask_name, resized_mask)




# # Understanding the annotation
# # getAnnIds return a list of all annotation which are associated with imgage_id
# annotation_ids = coco.getAnnIds(imgIds=img['id'])
# # loadAnns return a list of actual annotation
# annotations = coco.loadAnns(annotation_ids)
# # Plotting the annotations
# plt.imshow(I)
# plt.axis('off')
# coco.showAnns(annotations)
# plt.show()

# pylab.rcParams['figure.figsize'] = (1, 1)
# mask = np.zeros((300,300))
# for _idx, annotation in enumerate(annotations):
#     plt.subplot(len(annotations), 1, _idx+1)
#     rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
#     m = cocomask.decode(rle)
#     m = m.reshape((img['height'], img['width']))
#     mask = np.maximum(mask, m)
#     # plt.imshow(m)
#     # plt.show()
#     plt.imsave('1.png', mask)
#
# resized_img = cv2.resize(I, (256, 256), interpolation=cv2.INTER_NEAREST)
# resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
# cv2.imwrite('1_img.png',resized_img)
#
# test_mask = cv2.imread('1.png')
# test_mask = cv2.cvtColor(test_mask, cv2.COLOR_BGR2GRAY)
# test_mask = cv2.resize(test_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
# print(test_mask.shape)
# print(np.unique(test_mask))
# for i in range(test_mask.shape[0]):
#     for j in range(test_mask.shape[1]):
#         if test_mask[i,j] <=100:
#             test_mask[i,j] = 0
#         else:
#             test_mask[i,j] = 255
# cv2.imwrite('11.png',test_mask)
