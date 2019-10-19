import sys
import cv2
from keras.models import Model, load_model

model = load_model(sys.argv[1])
model.summary()
