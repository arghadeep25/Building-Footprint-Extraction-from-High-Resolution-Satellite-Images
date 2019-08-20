"""
Name: Fully Convolutional Network
Author: Arghadeep Mazumder
Version: 0.1
Description: FCN Architecture
"""

from keras.models import Model
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add
from keras.layers import Cropping2D
# from keras.layers.normalization import BatchNormalization
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
                    - Max Pool Strides (default = 2)
                    - Up Sample (default = 2)
    """
    def __init__(self,
                 classes=2,
                 image_size=256,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 pool_size=(2, 2),
                 padding='same',
                 activation='relu'):
        self.classes = classes
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.padding = padding
        self.activation = activation

    def network(self):
        # inputs
        inputs = Input((self.image_size, self.image_size, 3))
        # filters
        f = [64, 128, 256, 512, 4096]
        levels = []
        # VGG-16 Encoder
        # Block 1
        conv1 = Conv2D(f[0],
                       kernel_size=self.kernel_size,
                       activation=self.activation,
                       padding=self.padding)(inputs)
        conv2 = Conv2D(f[0],
                       kernel_size=self.kernel_size,
                       activation=self.activation,
                       padding=self.padding)(conv1)
        maxpool1 = MaxPooling2D(pool_size=(2, 2),
                                strides=self.strides)(conv2)
        maxpool1 = Dropout(0.5)(maxpool1)
        levels.append(maxpool1)

        # Block 2
        conv3 = Conv2D(f[1],
                       kernel_size=self.kernel_size,
                       activation=self.activation,
                       padding=self.padding)(maxpool1)
        conv4 = Conv2D(f[1],
                       kernel_size=self.kernel_size,
                       activation=self.activation,
                       padding=self.padding)(conv3)
        maxpool2 = MaxPooling2D(pool_size=self.pool_size,
                                strides=self.strides)(conv4)
        levels.append(maxpool2)

        # Block 3
        conv5 = Conv2D(f[2],
                       kernel_size=self.kernel_size,
                       activation=self.activation,
                       padding=self.padding)(maxpool2)
        conv6 = Conv2D(f[2],
                       kernel_size=self.kernel_size,
                       activation=self.activation,
                       padding=self.padding)(conv5)
        conv7 = Conv2D(f[2],
                       kernel_size=self.kernel_size,
                       activation=self.activation,
                       padding=self.padding)(conv6)
        maxpool3 = MaxPooling2D(pool_size=self.pool_size,
                                strides=self.strides)(conv7)
        levels.append(maxpool3)

        # Block 4
        conv8 = Conv2D(f[3],
                       kernel_size=self.kernel_size,
                       activation=self.activation,
                       padding=self.padding)(maxpool3)
        conv9 = Conv2D(f[3],
                       kernel_size=self.kernel_size,
                       activation=self.activation,
                       padding=self.padding)(conv8)
        conv10 = Conv2D(f[3],
                        kernel_size=self.kernel_size,
                        activation=self.activation,
                        padding=self.padding)(conv9)
        maxpool4 = MaxPooling2D(pool_size=self.pool_size,
                                strides=self.strides)(conv10)
        levels.append(maxpool4)

        # Block 5
        conv11 = Conv2D(f[3],
                        kernel_size=self.kernel_size,
                        activation=self.activation,
                        padding=self.padding)(maxpool4)
        conv12 = Conv2D(f[3],
                        kernel_size=self.kernel_size,
                        activation=self.activation,
                        padding=self.padding)(conv11)
        conv13 = Conv2D(f[3],
                        kernel_size=self.kernel_size,
                        activation=self.activation,
                        padding=self.padding)(conv12)
        maxpool5 = MaxPooling2D(pool_size=self.pool_size,
                                strides=self.strides)(conv13)
        levels.append(maxpool5)
        # --------------------------

        # conv14 = Conv2D(f[4],
        #                (7, 7),
        #                activation=self.activation,
        #                padding=self.padding)(levels[4])
        # conv15 = Conv2D(f[4],
        #                 (1, 1),
        #                 activation=self.activation,
        #                 padding=self.padding)(conv14)
        #
        # conv_tran1 = Conv2DTranspose(self.classes,
        #                              kernel_size=(4, 4),
        #                              strides=(4, 4),
        #                              use_bias=False)(conv15)
        # conv16 = Conv2D(self.classes,
        #                 (1, 1),
        #                 activation=self.activation,
        #                 padding=self.padding)(levels[3])
        # conv_tran2 = Conv2DTranspose(self.classes,
        #                              kernel_size=(2, 2),
        #                              strides=(2, 2),
        #                              use_bias=False)(conv16)
        # conv17 = Conv2D(self.classes,
        #                 (1, 1),
        #                 activation=self.activation,
        #                 padding=self.padding)(levels[2])
        #
        # sum1 = Add()([conv_tran2, conv17, conv_tran1])
        #
        # conv_tran3 = Conv2DTranspose(self.classes,
        #                              kernel_size=(8, 8),
        #                              strides=(8, 8),
        #                              use_bias=False)(sum1)
        # outputs = Activation('softmax')(conv_tran3)
        #
        # model = Model(inputs, outputs)
        #
        # return model

        # Converting connected layer to convolutional layer
        conv14 = Conv2D(f[4],
                        (7, 7),
                        activation=self.activation,
                        padding=self.padding)(levels[4])
        drop1 = Dropout(0.5)(conv14)
        conv15 = Conv2D(f[4],
                        (1, 1),
                        activation=self.activation,
                        padding=self.padding)(drop1)
        drop2 = Dropout(0.5)(conv15)

        conv16 = Conv2D(self.classes,
                        (1, 1),
                        kernel_initializer='he_normal')(drop2)
        conv_tran1 = Conv2DTranspose(self.classes,
                                     kernel_size=(4, 4),
                                     strides=self.strides,
                                     use_bias=False)(conv16)

        conv17 = Conv2D(self.classes,
                        (1, 1),
                        kernel_initializer='he_normal')(levels[3])
        # Cropping 1
        out_shape1 = Model(inputs, conv_tran1).output_shape
        out_shape1_height = out_shape1[1]
        out_shape1_width = out_shape1[2]
        print("Height 1",out_shape1_height)
        print("Width 1", out_shape1_width)

        out_shape2 = Model(inputs, conv17).output_shape
        out_shape2_height = out_shape2[1]
        out_shape2_width = out_shape2[2]
        print("Height 2",out_shape2_height)
        print("Width 2", out_shape2_width)

        diff1_w = abs(out_shape1_width - out_shape2_width)
        diff1_h = abs(out_shape2_height - out_shape1_height)
        print('Diff width',diff1_h)
        print('Diff height', diff1_w)

        if out_shape1_width > out_shape2_width:
            crop1 = Cropping2D(cropping=((0, diff1_w), (0, diff1_w)))(conv_tran1)
        else:
            crop1 = Cropping2D(cropping=((0, 0), (0, diff1_w)))(conv17)

        if out_shape1_height > out_shape2_height:
            crop2 = Cropping2D(cropping=((0, diff1_h), (0, diff1_h)))(conv_tran1)
        else:
            crop2 = Cropping2D(cropping=((0, diff1_h), (0, 0)))(conv17)

        sum1 = Add()([crop1, crop2])

        conv_tran2 = Conv2DTranspose(self.classes,
                                     kernel_size=(4, 4),
                                     strides=self.strides,
                                     use_bias=False)(sum1)
        conv18 = Conv2D(self.classes,
                        (1, 1),
                        kernel_initializer='he_normal')(levels[2])

        # Cropping 2 o1 = conv18 o2 = conv_tran2
        out_shape3 = Model(inputs, conv18).output_shape
        out_shape3_height = out_shape3[1]
        out_shape3_width = out_shape3[2]

        out_shape4 = Model(inputs, conv18).output_shape
        out_shape4_height = out_shape4[1]
        out_shape4_width = out_shape4[2]

        diff2_w = abs(out_shape3_width - out_shape4_width)
        diff2_h = abs(out_shape4_height - out_shape3_height)

        if out_shape3_width > out_shape4_width:
            crop3 = Cropping2D(cropping=((0, 0), (0, diff2_w)))(conv18)
        else:
            crop3 = Cropping2D(cropping=((0, 0), (0, diff2_w)))(conv_tran2)

        if out_shape3_height > out_shape4_height:
            crop4 = Cropping2D(cropping=((0, diff2_h), (0, 0)))(conv18)
        else:
            crop4 = Cropping2D(cropping=((0, diff2_h), (0, 0)))(conv_tran2)

        sum2 = Add()([crop3, crop4])

        conv_tran3 = Conv2DTranspose(self.classes,
                                     kernel_size=(16, 16),
                                     strides=(8, 8),
                                     use_bias=False)(sum2)

        # Reshaping Model
        output_shape = Model(inputs, conv_tran3).output_shape
        input_shape = Model(inputs, conv_tran3).input_shape

        output_shape_height = output_shape[1]
        output_shape_width = output_shape[2]
        out_classes = output_shape[3]

        reshape1 = Reshape((output_shape_height *
                            output_shape_width, -1))(conv_tran3)

        outputs = Activation('softmax')(reshape1)

        model = Model(inputs, outputs)

        return model
