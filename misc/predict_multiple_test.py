import cv2
import os
import numpy as np
import sys
from keras.models import load_model
from keras.preprocessing.image import img_to_array

model = load_model(sys.argv[1])
folder_path = sys.argv[2]
output_folder = sys.argv[3]
image_path = os.path.join(folder_path, 'images/')

_, _, files = next(os.walk(image_path))
result = np.zeros((256, 256, 3))

for file in files:
    image_name = image_path + file
    image = cv2.imread(image_name)
    image = cv2.resize(image, (256,256), interpolation=cv2.INTER_NEAREST)
    original_image = image.astype(np.uint8)
    image = img_to_array(image)/255.
    # image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    # image[:,:,1] = cv2.equalizeHist(image[:,:,1])
    # image[:,:,2] = cv2.equalizeHist(image[:,:,2])
    # kernel_sharp = np.array([[-1, -1, -1],
    #                          [-1, 9, -1],
    #                          [-1, -1, -1]])
    # image = cv2.filter2D(image, -1, kernel_sharp)
    image = np.expand_dims(image, 0)
    prediction = model.predict(image)

    prediction = prediction > 0.5

    prediction = np.reshape(prediction[0]*255,(256, 256))
    prediction = prediction.astype(np.uint8)
    # print(prediction)
    result[:,:,2] = prediction
    print(result[:,:,2])
    result = result.astype(np.uint8)

    overlay = cv2.addWeighted(result, 1, original_image, 1, 0)

    cv2.imwrite(output_folder+file, overlay)
