import os
import sys
import time
import cv2
from utils import loader

def main():
    path = sys.argv[1]

    data = loader.InriaDataLoader(path, rotation =90, aug = True)
    images, masks = data.__getitem__()

    i =30
    cv2.imshow('Image', images[i])
    cv2.imshow('Mask', masks[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(len(images))

if __name__ == '__main__':
    main()
