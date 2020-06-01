"""
Name: U-Net Model
Author: Arghadeep Mazumder
Version: 0.1
Description: U-Net Architecture
"""
import keras
from typing import List, Tuple
from keras.models import Model, load_model
from keras.layers import Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate

class UNet():
    """ U-Net Model

        Parameters: - Image Size (default = 256)
                    - Kernel Size (default = (3, 3))
                    - Padding (default = 'same')
                    - Activation (default = 'relu')
                    - Pool Size (default = 2)
                    - Strides (default = 1)
                    - Max Pool Strides (default = 2)
                    - Up Sample (default = 2)
    """
    def __init__(self,
                image_size: int = 256,
                kernel_size: Tuple[int] = (3, 3),
                padding: str = 'same',
                activation: str = 'relu',
                pool_size: int = 2,
                strides: int = 1,
                max_pool_strides: int = 2,
                up_sample: int = 2,
                pre_trained: bool = False) -> None:
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.pool_size = pool_size
        self.strides = strides
        self.max_pool_strides = max_pool_strides
        self.up_sample = up_sample
        self.pre_trained = pre_trained


    def network(self) -> keras.models.Model:
        f = [16, 32, 64, 128, 256]
        inputs = Input((self.image_size, self.image_size, 3))

        # p0 = inputs
        down1 = Conv2D(f[0],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(inputs)
        down2 = Conv2D(f[0],
                    kernel_size= self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation = self.activation)(down1)
        pool1 = MaxPool2D(pool_size = (self.pool_size, self.pool_size),
                    strides = (self.max_pool_strides,
                            self.max_pool_strides))(down2)

        down3 = Conv2D(f[1],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(pool1)
        down4 = Conv2D(f[1],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(down3)
        pool2 = MaxPool2D(pool_size = (self.pool_size, self.pool_size),
                    strides = (self.max_pool_strides,
                            self.max_pool_strides))(down4)

        down5 = Conv2D(f[2],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation = self.activation)(pool2)
        down6 = Conv2D(f[2],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(down5)
        pool3 = MaxPool2D(pool_size = (self.pool_size, self.pool_size),
                    strides = (self.max_pool_strides,
                            self.max_pool_strides))(down6)

        down7 = Conv2D(f[3],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(pool3)
        down8 = Conv2D(f[3],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(down7)
        pool4 = MaxPool2D(pool_size = (self.pool_size, self.pool_size),
                    strides = (self.max_pool_strides,
                            self.max_pool_strides))(down8)

        bn1 = Conv2D(f[4],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(pool4)
        bn2 = Conv2D(f[4],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(bn1)

        up_sample1 = UpSampling2D((self.up_sample, self.up_sample))(bn2)

        concat1 = Concatenate()([up_sample1, down8])

        up1 = Conv2D(f[3],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(concat1)
        up2 = Conv2D(f[3],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(up1)

        up_sample2 = UpSampling2D((self.up_sample, self.up_sample))(up2)

        concat2 = Concatenate()([up_sample2, down6])

        up3 = Conv2D(f[2],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(concat2)
        up4 = Conv2D(f[2],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(up3)

        up_sample3 = UpSampling2D((self.up_sample, self.up_sample))(up4)

        concat3 = Concatenate()([up_sample3, down4])

        up5 = Conv2D(f[1],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(concat3)
        up6 = Conv2D(f[1],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(up5)

        up_sample4 = UpSampling2D((self.up_sample, self.up_sample))(up6)

        concat4 = Concatenate()([up_sample4, down2])

        up7 = Conv2D(f[0],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(concat4)
        up8 = Conv2D(f[0],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    strides= self.strides,
                    activation= self.activation)(up7)

        outputs = Conv2D(1, (1,1), padding = 'same',
                        activation= 'sigmoid')(up8)
        model = Model(inputs, outputs)

        if self.pre_trained is True:
            model = load_model('../trained_models/unet_inria.h5')

        return model
