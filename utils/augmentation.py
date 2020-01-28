"""
Name: Data Augmentation
Author: Arghadeep Mazumder
Version: 0.1
Description: 1. Rotate
             2. Rescale
             3. Flip Horizontally
             4. Flip Vertically
             5. Shear
             6. Histogram Equalization
             7. Brightness
             8. Noise
             9. Smoothing
"""
import cv2
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize, rotate, rescale, warp, AffineTransform


class DataAugmentation():
    """ Class for data augmentation.

        Mainly helps when the training data is small

        Parameters: image, mask,
                    rotation angle (default = 90)
                    zoom_range (default = 1)
                    horizontal_flip (default = False)
                    vertical_flip (default = False)
                    activate (default = False)
    """

    def __init__(self, image, mask,
                rotation = 0,
                zoom_range = 1,
                horizontal_flip = False,
                vertical_flip = False,
                shear = 0,
                hist_eq = False,
                brightness = False,
                add_noise = False,
                sigma = 0,
                activate = False):

        self.image = image
        self.mask = mask
        self.rotation = rotation
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.shear = shear
        self.hist_eq = hist_eq
        self.brightness = brightness
        self.add_noise = add_noise
        self.sigma = sigma
        self.activate = activate

    def rotate_data(self):
        """ Rotation
        """
        image = self.image
        mask = self.mask
        image = rotate(image, self.rotation)
        mask = rotate(mask, self.rotation)
        return image/255., mask/255.

    def zoom_func(self, image):
        """Zoom Function
        """
        if len(image.shape) == 3:
            height, width, channel = image.shape
        else:
            height, width = image.shape

        zoom_tuple = (self.zoom_range,)*2 + (1,)*(image.ndim - 2)

        z_height = int(np.round(height / self.zoom_range))
        z_width = int(np.round(height / self.zoom_range))
        top = (height - z_height) // 2
        left = (width - z_width) // 2

        out = zoom(image[top:top+z_height, left:left+z_width], zoom_tuple)

        trim_top = ((out.shape[0] - height) // 2)
        trim_left = ((out.shape[1] - width) // 2)
        out = out[trim_top:trim_top+height, trim_left:trim_left+width]
        return out

    def rescale_data(self):
        """ Clipped Zoom
        """
        image = self.image
        mask = self.mask

        image = self.zoom_func(image)
        mask = self.zoom_func(mask)

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j] < 200:
                    mask[i,j] = 0
                else:
                    mask[i,j] = 255
        return image/255., mask/255.

    def flip_horizontal_data(self):
        """ Flip Horizontally
        """
        image = self.image
        mask = self.mask
        if self.horizontal_flip == True:
            flipped_image = np.flip(image, 1)
            flipped_mask = np.flip(mask, 1)
            return flipped_image/255., flipped_mask/255.

    def flip_vertically_data(self):
        """ Flip Vertically
        """
        image = self.image
        mask = self.mask
        if self.vertical_flip == True:
            flipped_image = np.flip(image, 0)
            flipped_mask = np.flip(mask, 0)
            return flipped_image/255., flipped_mask/255.

    def shear_data(self):
        """ Shear
        """
        image = self.image
        mask = self.mask
        trans = AffineTransform(shear = 0.2)
        image = warp(image, inverse_map= trans)
        mask = warp(mask, inverse_map= trans)
        return image/255., mask/255.

    def histogram_equalization(self):
        """ Histogram Equalization
        """
        image = self.image
        image = image.astype(np.uint8)
        image[:,:,0] = cv2.equalizeHist(image[:,:,0])
        image[:,:,1] = cv2.equalizeHist(image[:,:,1])
        image[:,:,2] = cv2.equalizeHist(image[:,:,2])

        image = image/255.
        return image, self.mask/255.

    def add_brightness(self):
        """Brightness Level
        """
        image = self.image
        image[:,:,0] = image[:,:,0] * 0.5
        image[:,:,1] = image[:,:,1] * 0.5
        image[:,:,2] = image[:,:,2] * 0.5
        return image/255., self.mask/255.

    def gaussian_noise(self):
        """Gaussian Noise
        """
        image = self.image
        height, width, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5

        gauss = np.random.normal(mean,sigma,(height, width, ch))
        gauss = gauss.reshape(height, width, ch)*80

        image[:,:,0] = image[:,:,0] + gauss[:,:,0]
        image[:,:,1] = image[:,:,1] + gauss[:,:,1]
        image[:,:,2] = image[:,:,2] + gauss[:,:,2]
        return image/255., self.mask/255.

    def smooth(self):
        """Blurring image
        """
        image = self.image
        kernel = (2*self.sigma + 1)
        blurred = cv2.GaussianBlur(image, (kernel, kernel), cv2.BORDER_CONSTANT)
        return blurred/255., self.mask/255.

    def augment(self):
        """Activate augmentations based on parameters
        """
        if self.activate == True:
            images = []
            masks = []
            images.append(self.image/255.)
            masks.append(self.mask/255.)
            # print('Augmentation:: Image List Size: ',len(images))
            if self.rotation != 0:
                rotate_image, rotate_mask = self.rotate_data()
                images.append(rotate_image)
                masks.append(rotate_mask)

            if self.zoom_range != 1:
                zoom_image, zoom_mask = self.rescale_data()
                images.append(self.image)
                masks.append(self.mask)

            if self.horizontal_flip == True:
                horflip_image, horflip_mask = self.flip_horizontal_data()
                images.append(horflip_image)
                masks.append(horflip_mask)

            if self.vertical_flip == True:
                verflip_image, verflip_mask = self.flip_vertically_data()
                images.append(verflip_image)
                masks.append(verflip_mask)

            if self.shear != 0:
                shear_image, shear_mask= self.shear_data()
                images.append(shear_image)
                masks.append(shear_mask)

            if self.hist_eq is True:
                histeq_image, histeq_mask= self.histogram_equalization()
                images.append(histeq_image)
                masks.append(histeq_mask)

            if self.brightness is True:
                bright_image, bright_mask = self.add_brightness()
                images.append(bright_image)
                masks.append(bright_mask)

            if self.add_noise is True:
                noise_image, noise_mask = self.gaussian_noise()
                images.append(noise_image)
                masks.append(noise_mask)

            if self.sigma > 0:
                smooth_image, smooth_gt = self.smooth()
                images.append(smooth_image)
                masks.append(smooth_gt)

        else:
            images = []
            masks = []
            images.append(self.image/255.)
            masks.append(self.mask/255.)

        images = np.array(images)
        masks = np.array(masks)

        return images, masks
