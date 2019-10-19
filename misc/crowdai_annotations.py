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

# Loading annotation path into memory
coco = COCO(train_annotation_small_path)

category_ids = coco.loadCats(coco.getCatIds())

# collecting and visualizing images
# Generating lists of all images
image_ids = coco.getImgIds(catIds=coco.getCatIds())
# Randomly picking an image
random_image_id = random.choice(image_ids)

# Displaying a random image
img = coco.loadImgs(random_image_id)[0]
image_path = os.path.join(train_images_dir, img['file_name'])
print(image_path)
I = io.imread(image_path)
# plt.imshow(I)
# plt.show()

# Understanding the annotation
# getAnnIds return a list of all annotation which are associated with imgage_id
annotation_ids = coco.getAnnIds(imgIds=img['id'])
# loadAnns return a list of actual annotation
annotations = coco.loadAnns(annotation_ids)
# print('Annotations',annotations)
# Plotting the annotations
fig = plt.imshow(I)
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
coco.showAnns(annotations)
plt.savefig('ann_5.png', bbox_inches='tight', pad_inches=0)
plt.show()

annotate_mask = cv2.imread('ann_5.png')
annotate_mask = cv2.resize(annotate_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
cv2.imwrite('ann_6.png', annotate_mask)
# m = m.reshape((img['height'], img['width']))
# plt.imshow(m)
# plt.show()

# pylab.rcParams['figure.figsize'] = (1, 1)
mask = np.zeros((300,300))
for _idx, annotation in enumerate(annotations):
    # plt.subplot(len(annotations), 1, _idx+1)
    rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
    m = cocomask.decode(rle)
    m = m.reshape((img['height'], img['width']))
    mask = np.maximum(mask, m)
    # plt.imshow(m)
    # plt.show()
    plt.imsave('1.png', mask)

resized_img = cv2.resize(I, (256, 256), interpolation=cv2.INTER_NEAREST)
resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
cv2.imwrite('6_img.png',resized_img)

test_mask = cv2.imread('1.png')
test_mask = cv2.cvtColor(test_mask, cv2.COLOR_BGR2GRAY)
test_mask = cv2.resize(test_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
print(test_mask.shape)
print(np.unique(test_mask))
for i in range(test_mask.shape[0]):
    for j in range(test_mask.shape[1]):
        if test_mask[i,j] <=100:
            test_mask[i,j] = 0
        else:
            test_mask[i,j] = 255
cv2.imwrite('66.png',test_mask)
