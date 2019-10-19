import cv2
import sys
import os

image_path = sys.argv[1]
save_dir_path = sys.argv[2]
_, _, files = next(os.walk(image_path))

for file in files:
    name = os.path.splitext(file)[0]
    load_image_name = os.path.join(image_path, file)
    image = cv2.imread(load_image_name)
    image_name = save_dir_path + name + '.png'
    cv2.imwrite(image_name, image)
