import numpy as np
import cv2
from keras.models import Model, load_model
import sys
from keras.preprocessing.image import img_to_array

image = cv2.imread(sys.argv[1])
print(image.dtype)
print(np.max(image))
or_img = image
image = cv2.resize(image, (256, 256))

# image[:,:,0] = cv2.equalizeHist(image[:,:,0])
# image[:,:,1] = cv2.equalizeHist(image[:,:,1])
# image[:,:,2] = cv2.equalizeHist(image[:,:,2])
# kernel_sharp = np.array([[-1, -1, -1],
#                          [-1, 9, -1],
#                          [-1, -1, -1]])
#
# image = cv2.filter2D(image, -1, kernel_sharp)

or_img = or_img.astype(np.uint8)
image = img_to_array(image)/255.
image = np.expand_dims(image, 0)
model = load_model(filepath=sys.argv[2])

# model.summary()

prediction = model.predict(image)
prediction = prediction > 0.5
print("Prediction Shape: ",prediction.shape)
print('Prediction Shape 0: ', prediction[0].shape)
prediction = np.reshape(prediction[0]*255, (256, 256))
pred = prediction
prediction = prediction.astype(np.uint8)

gt = cv2.imread(sys.argv[3])
gt = cv2.resize(gt, (256, 256))
gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

intersection = np.logical_and(gt, prediction)
union = np.logical_or(gt, prediction)
iou_score = np.sum(intersection)/ np.sum(union)
print('IoU Score: ',iou_score)

result = np.zeros((256, 256, 3))

result[:,:,1] = prediction
result = result.astype(np.uint8)

overlay = cv2.addWeighted(result,1, or_img, 1, 0)

# cv2.imwrite('results/segnet_epoch30_bs8_austin1_3_mask.png',pred)
# cv2.imwrite('results/segnet_epoch30_bs8_austin1_3_res.png', overlay)
cv2.imshow('Result',overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print(np.unique(prediction))
# print(prediction.shape)
