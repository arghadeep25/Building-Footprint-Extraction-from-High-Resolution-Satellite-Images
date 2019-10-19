import os
import cv2
import numpy as np

folder_path = 'austin/gt/'
_, _, files = next(os.walk(folder_path))

nof = len(files)
nof = nof - 1
result = np.zeros((256, 256*20, 3))
print(result.shape)
for i in range(nof):
    filename = folder_path+'austin1_' + str(i)+'.tif'
    img = cv2.imread(filename)
    print(filename)
    np.hstack(())
