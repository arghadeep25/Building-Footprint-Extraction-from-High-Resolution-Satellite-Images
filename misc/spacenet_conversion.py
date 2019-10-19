import cv2
import sys
import rasterio
import numpy as np

image = sys.argv[1]

test = np.zeros((650, 650, 3))
cv2.imwrite('tst.jpg',test)
with rasterio.open(image) as src:
    data = src.read()
    profile = src.profile

minBand = np.percentile(data, 0)
maxBand = np.percentile(data, 98)

scale_ratio = 255/maxBand

data = data.astype('float64', casting='unsafe',copy=False)

data[data>=maxBand] = maxBand
data = data*scale_ratio

data = data.astype('uint8', casting='unsafe',copy=False)

profile['driver'] = 'GTiff'
profile['dtype'] = 'uint8'
del profile['tiled']
del profile['interleave']

outputImageName = 'tst.jpg'
with rasterio.open(outputImageName, 'w', **profile) as dst:
    dst.write(data)
