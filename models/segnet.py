"""
Name: SegNet Model
Author: Arghadeep Mazumder
Version: 0.1
Description: SegNet Architecture
"""
import keras
from typing import Tuple
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Reshape
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling2D, UpSampling2D


class SegNet:
    def __init__(self,image_size: int = 256,
                 kernel_size: Tuple[int] = (3, 3),
                 padding: str = 'same',
                 activation: str = 'relu',
                 pool_size: int = 2,
                 strides: int = 1,
                 max_pool_strides: int = 2,
                 up_sample: int = 2,
                 pre_trained: bool = False):
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

        img_input = Input((self.image_size, self.image_size, 3))
        f = [64, 128, 256, 512]

        x = img_input
        # Encoder
        x = Conv2D(f[0],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv2D(f[0],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(x)

        x = Conv2D(f[1],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv2D(f[1],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(x)

        x = Conv2D(f[2],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv2D(f[2],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(x)

        x = Conv2D(f[3],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv2D(f[3],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv2D(f[3],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding = self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(x)

        x = Conv2D(f[3],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv2D(f[3],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv2D(f[3],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding = self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(x)

        # Decoder
        x = UpSampling2D(size=(self.up_sample, self.up_sample))(x)
        x = Conv2D(f[3],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv2D(f[3],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv2D(f[3],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)

        x = UpSampling2D(size=(self.up_sample, self.up_sample))(x)
        x = Conv2D(f[3],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv2D(f[3],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv2D(f[2],
                   kernel_size = self.kernel_size,
                   strides=(self.strides,self.strides),
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)

        x = UpSampling2D(size=(self.up_sample, self.up_sample))(x)
        x = Conv2D(f[2],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv2D(f[2],
                   kernel_size = self.kernel_size,
                   strides=(self.strides,self.strides),
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv2D(f[1],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)

        x = UpSampling2D(size=(self.up_sample, self.up_sample))(x)
        x = Conv2D(f[1],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv2D(f[0],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)

        x = UpSampling2D(size=(self.up_sample, self.up_sample))(x)
        x = Conv2D(f[0],
                   kernel_size = self.kernel_size,
                   strides = self.strides,
                   padding=self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Conv2D(2,
                   (1,1),
                   strides=(1, 1),
                   padding='valid')(x)
        x = BatchNormalization()(x)

        x = Reshape((self.image_size*self.image_size, 2))(x)
        x = Activation("softmax")(x)
        model = Model(img_input, x)

        if self.pre_trained is True:
            model = load_model('../trained_models/segnet_inria.h5')

        return model
