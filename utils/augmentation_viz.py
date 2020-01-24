import cv2
from augmentation import DataAugmentation

img_ip = cv2.imread('austin1_50.png', 3)
mask_ip = cv2.imread('austin1_gt_50.png')

aug = DataAugmentation(image=img_ip,
                       mask=mask_ip,
                       rotation= 135,
                       zoom_range=1.5,
                       horizontal_flip= True,
                       vertical_flip= True,
                       shear= 0.1,
                       hist_eq=True,
                       brightness=True,
                       add_noise=True,
                       activate=True)

img_rot, mask_rot = aug.rotate_data()
cv2.imwrite('austin1_50_rot.png',img_rot*255*255)
cv2.imwrite('austin1_gt_50_rot.png',mask_rot*255*255)

# img_scale, mask_scale = aug.rescale_data()
# cv2.imwrite('austin1_50_scale.png', img_scale*255)
# cv2.imwrite('austin1_gt_50_scale.png', mask_scale*255)

# img_hf, mask_hf = aug.flip_horizontal_data()
# cv2.imwrite('austin1_50_hf.png', img_hf*255)
# cv2.imwrite('austin1_gt_50_hf.png', mask_hf*255)
#
# img_vf, mask_vf = aug.flip_vertically_data()
# cv2.imwrite('austin1_50_vf.png', img_vf*255)
# cv2.imwrite('austin1_gt_50_vf.png',mask_vf*255)
#
# img_sh, mask_sh = aug.shear_data()
# cv2.imwrite('austin1_50_sh.png', img_sh*255*255)
# cv2.imwrite('austin1_gt_50_sh.png', mask_sh*255*255)
#
# img_hist, mask_hist = aug.histogram_equalization()
# cv2.imwrite('austin1_50_hist.png',img_hist*255)
# cv2.imwrite('austin1_gt_50_hist.png',mask_hist*255)
#
# img_bright, mask_bright = aug.add_brightness()
# cv2.imwrite('austin1_50_bright.png',img_bright*255)
# cv2.imwrite('austin1_gt_50_bright.png',mask_bright*255)
#
# img_noise, mask_noise = aug.gaussian_noise()
# cv2.imwrite('austin1_50_noise.png',img_noise*255)
# cv2.imwrite('austin1_gt_50_noise.png',mask_noise*255)
# cv2.imwrite()
print('success')
