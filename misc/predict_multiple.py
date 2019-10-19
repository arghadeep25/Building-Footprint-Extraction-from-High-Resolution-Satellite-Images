import numpy as np
import cv2
from keras.models import Model, load_model
from keras.preprocessing.image import img_to_array
import sys
import os

# def predict_image(image, model):
def iou(prediction, gt):
    intersection = np.logical_and(gt, prediction)
    union = np.logical_or(gt, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def extract_masks(mask, cl):
    new_mask = np.zeros((2,256,256))
    for i, c in enumerate(cl):
        new_mask[i,:,:] = mask == c
    return new_mask

def pixel_accuracy(prediction, gt):
    cl = np.unique(gt)
    n_cl = len(cl)
    pred_mask = extract_masks(prediction, cl)
    gt_mask = extract_masks(gt, cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_pred_mask = pred_mask[i,:,:]
        curr_gt_mask = gt_mask[i,:,:]

        sum_n_ii += np.sum(np.logical_and(curr_pred_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)
    if sum_t_i == 0:
        pixel_accuracy_val = 0
    else:
        pixel_accuracy_val = sum_n_ii/sum_t_i
    return pixel_accuracy_val


def dice_coeff(prediction, gt):
    dice = np.sum(prediction[gt==255])*2.0 / (np.sum(prediction) + np.sum(gt))
    return dice

model = load_model(sys.argv[1])
folder_path = sys.argv[2]
output_folder = sys.argv[3]
image_path = os.path.join(folder_path, 'images/')
mask_path = os.path.join(folder_path, 'gt/')
_, _, files = next(os.walk(image_path))

result = np.zeros((256, 256, 3))
avg_iou = []
avg_dice = []
avg_pixel_acc = []

for file in files:
    name = os.path.splitext(file)[0]
    image_name = image_path + file
    mask_name = mask_path + file
    image = cv2.imread(image_name)
    # cv2.imwrite(output_folder+'images/'+name+'.png',image)
    original_image = image.astype(np.uint8)
    original_image = cv2.resize(original_image, (256, 256))
    image = cv2.resize(image, (256, 256))
    image = img_to_array(image)/255.
    image = np.expand_dims(image, 0)
    print(image_name)
    mask = cv2.imread(mask_name)
    # cv2.imwrite(output_folder+'gt/'+name+'.png',mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = cv2.resize(mask, (256, 256))
    prediction = model.predict(image)
    prediction = prediction > 0.5
    prediction = np.reshape(prediction[0]*255, (256, 256))
    prediction = prediction.astype(np.uint8)
    iou_score = iou(prediction, mask)
    dice_score = dice_coeff(prediction, mask)
    pixel_accuracy_score = pixel_accuracy(prediction, mask)

    if iou_score > 0.2:
        avg_iou.append(iou_score)
    if dice_score > 0.2:
        avg_dice.append(dice_score)
    if pixel_accuracy_score > 0.2:
        avg_pixel_acc.append(pixel_accuracy_score)
    result[:, :, 2] = prediction
    result = result.astype(np.uint8)
    overlay = cv2.addWeighted(result, 1, original_image, 1, 0)

    cv2.imwrite(output_folder +name +'.png', overlay)

avg_score = sum(avg_iou)/len(avg_iou)
print('Average IoU: ',avg_score)
avg_dice_score = sum(avg_dice)/len(avg_dice)
print('Average Dice Score: ',avg_dice_score)
avg_pixel_score = sum(avg_pixel_acc)/len(avg_pixel_acc)
print('Average Pixel Accuracy: ',avg_pixel_score)
