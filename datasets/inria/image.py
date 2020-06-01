import cv2
import os
import sys

path = sys.argv[1]
_, _, files = next(os.walk(path))
print(files)

for file in files:
    name = os.path.splitext(file)[0]
    image_path = os.path.join(path, file)
    image = cv2.imread(image_path)
    filename = path + name + '.png'
    # print(image)
    cv2.imwrite(filename, image)
