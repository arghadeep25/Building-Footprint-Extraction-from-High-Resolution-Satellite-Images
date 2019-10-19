import gdal
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt

data_root = sys.argv[1]

mul_ds = gdal.Open(data_root)
channels = mul_ds.RasterCount
mul_img = np.zeros((mul_ds.RasterXSize, mul_ds.RasterYSize, channels), dtype=np.uint16)

geoTf = np.asarray(mul_ds.GetGeoTransform())

bmin = []
bmax = []
for band in range(0, channels):
    mul_img[:,:,band] = mul_ds.GetRasterBand(band+1).ReadAsArray()
    min = np.min(mul_img[:,:,band])
    max = np.max(mul_img[:,:,band])
    # print(max)
    bmin.append(min)
    bmax.append(max)

height, width, channel = mul_img.shape

# norm_img = np.zeros((height, width, channel))

# for i in range(3):
    # norm_img[:,:,i] = mul_img[:,:,i] * (255/bmax[i])

mul_img[:,:,0] = cv2.normalize(mul_img[:,:,0], dst=None, alpha=0, beta = 255, norm_type=cv2.NORM_MINMAX)
mul_img[:,:,1] = cv2.normalize(mul_img[:,:,1], dst=None, alpha=0, beta = 255, norm_type=cv2.NORM_MINMAX)
mul_img[:,:,2] = cv2.normalize(mul_img[:,:,2], dst=None, alpha=0, beta = 255, norm_type=cv2.NORM_MINMAX)
# norm_img = norm_img.astype(np.uint8)

# color = ('b', 'g', 'r')
# for i, col in enumerate(color):
#     histr = cv2.calcHist([mul_img],[i], None, [65535], [0,65535])
#     plt.plot(histr,color = col)
#     plt.xlim([0, 65535])
# plt.show()
mul_img = mul_img.astype(np.uint8)
# cv2.imshow('Image',mul_img)
cv2.imshow('Image',mul_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Khartoum38_gdal.png',mul_img)
# print(norm_img[:,:,0])
# mul_img = cv2.cvtColor(mul_img, cv2.COLOR_BGR2RGB)
# # mul_img = cv2.cvtColor(mul_img, cv2.COLOR_RGB2BGR)
# mul_img = cv2.normalize(mul_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
# mul_img = (mul_img).astype(np.uint8)
# print(mul_img.shape)
# print(np.unique(mul_img[:,:,1]))
# cv2.imshow('image',mul_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('image.tif',mul_img)

# import numpy as np
# import cv2
# import sys
# import gdal
#
# image = gdal.Open(sys.argv[1])
#
# for i in range(image.RasterCount):
#     i = i + 1
#     band = image.GetRasterBand(i)
#     # bmin = band.GetMinimum()
#     # bmax = band.GetMaximum()
#     (bmin, bmax) = band.ComputeRasterMinMax(1)
#     band_arr_tmp = band.ReadAsArray()
#     # bmin = np.percentile(band_arr_tmp.flatten(), 2)
#     # bmax = np.percentile(band_arr_tmp.flatten(), 90)
#     f_bmin = bmin * (255/bmax)
#     f_bmax = bmax * (255/bmax)
#     print('bmax: ',f_bmax)
