"""
Name: Fully Convolutional Network
Author: Arghadeep Mazumder
Version: 0.1
Description: FCN Architecture
"""

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add
from keras.layers import Cropping2D
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from keras.layers import Dropout


class FCN():
    """ Fully Convolutional Network Model

        Parameters: - Image Size (default = 256)
                    - Kernel Size (default = (3, 3))
                    - Padding (default = 'same')
                    - Activation (default = 'relu')
                    - Pool Size (default = 2)
                    - Strides (default = 1)
                    - Pre-Trained (default = False)
    """
    def __init__(self,
                 image_size=256,
                 kernel_size=(3, 3),
                 strides=1,
                 pool_size=2,
                 padding='same',
                 activation='relu',
                 pre_trained=False):
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.padding = padding
        self.activation = activation
        self.pre_trained = pre_trained

    def network(self):
        inputs = Input((self.image_size,self.image_size,3))
        filter = [32, 64, 128]

        c1 = Conv2D(filter[1],
                    kernel_size = self.kernel_size,
                    strides = self.strides,
                    activation=self.activation,
                    padding=self.padding)(inputs)

        c2 = Conv2D(filter[1],
                    kernel_size = self.kernel_size,
                    strides = self.strides,
                    activation=self.activation,
                    padding=self.padding)(c1)

        mp1 = MaxPooling2D((self.pool_size,self.pool_size))(c2)

        c3 = Conv2D(filter[2],
                    kernel_size = self.kernel_size,
                    strides = self.strides,
                    activation=self.activation,
                    padding=self.padding)(mp1)

        mp2 = MaxPooling2D((self.pool_size,self.pool_size))(c3)

        c4 = Conv2D(filter[2],
                    kernel_size = self.kernel_size,
                    strides = self.strides,
                    activation=self.activation,
                    padding=self.padding)(mp2)

        d1 = Conv2DTranspose(filter[1],
                             kernel_size = self.kernel_size,
                             activation=self.activation,
                             padding=self.padding,
                             strides=self.strides)(c4)

        m1 = concatenate([d1,c3])

        d2 = Conv2DTranspose(filter[0],
                             kernel_size = self.kernel_size,
                             activation=self.activation,
                             padding=self.padding,
                             strides=self.strides)(m1)

        m2 = concatenate([d2,c2])

        c5 = Conv2D(filter[1],
                    kernel_size = self.kernel_size,
                    strides = self.strides,
                    activation=self.activation,
                    padding=self.padding)(m2)

        c6 = Conv2D(1,
                    kernel_size = self.kernel_size,
                    strides = self.strides,
                    activation='sigmoid',
                    padding=self.padding)(c5)

        model = Model(inputs=i,outputs=c6)

        if self.pre_trained is True:
            model = load_model('../trained_models/fcn_inria.h5')

        return model
