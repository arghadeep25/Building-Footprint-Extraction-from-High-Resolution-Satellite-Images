"""
Name: Pyramid Scene Parsing Network (PSPNet) Model
Author: Arghadeep Mazumder
Version: 0.1
Description: PSPNet Architecture with VGG16 backbone
"""
import keras
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from keras.models import Model
from keras.layers import Input
from keras import backend as K
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add, Concatenate

class PSPNet():
    """ Pyramid Scene Parsing Network Model

        Parameters: - Image Size (default = 384)
                    - Kernel Size (default = (3, 3))
                    - Padding (default = 'same')
                    - Activation (default = 'relu')
                    - Pool Size (default = 2)
                    - Strides (default = 1)
                    - Max Pool Strides (default = 2)
                    - Up Sample (default = 2)
                    - Pre-trained (default = False)
    """
    def __init__(self, image_size: int = 384,
                 kernel_size: Tuple[int] = (3, 3),
                 n_classes: int = 2,
                 padding: str = 'same',
                 activation: str = 'relu',
                 pool_size: int = 2,
                 strides: int = 1,
                 max_pool_strides: int = 2,
                 up_sample: int = 2,
                 pre_trained: bool = False) -> None:
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.n_classes = n_classes
        self.padding = padding
        self.activation = activation
        self.pool_size = pool_size
        self.strides = strides
        self.max_pool_strides = max_pool_strides
        self.up_sample = up_sample
        self.pre_trained=pre_trained

    def network(self) -> keras.models.Model:
        # filters
        f = [64, 128, 256, 512, 1024, 2048]
        # pooling factors
        pool_factor = [1, 2, 3, 6]
        # inputs
        inputs = Input((self.image_size, self.image_size, 3))
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
                                strides=(2, 2))(conv2)
        maxpool1 = Dropout(0.5)(maxpool1)
        level1 = maxpool1
        # Block 2
        conv3 = Conv2D(f[1],
                       kernel_size=self.kernel_size,
                       activation=self.activation,
                       padding=self.padding)(maxpool1)
        conv4 = Conv2D(f[1],
                       kernel_size=self.kernel_size,
                       activation=self.activation,
                       padding=self.padding)(conv3)
        maxpool2 = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(conv4)
        maxpool2 = Dropout(0.5)(maxpool2)
        level2 = maxpool2
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
        maxpool3 = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(conv7)
        maxpool3 = Dropout(0.5)(maxpool3)
        level3 = maxpool3
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
        maxpool4 = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(conv10)
        maxpool4 = Dropout(0.5)(maxpool4)
        level4 = maxpool4
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
        maxpool5 = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(conv13)
        maxpool5 = Dropout(0.5)(maxpool5)
        level5 = maxpool5

        # Pyramid Pooling
        pool_output = [maxpool5]
        # Pyramid Block 1
        height_b1 = K.int_shape(level5)[1]
        width_b1 = K.int_shape(level5)[2]

        pool_size_b1 = [int(np.round(float(height_b1) / pool_factor[0])),
                        int(np.round(float(width_b1) / pool_factor[0]))
                        ]
        strides_b1 = pool_size_b1
        avg_pool1 = AveragePooling2D(pool_size_b1,
                                     strides=strides_b1,
                                     padding=self.padding)(level5)
        conv14 = Conv2D(f[3],
                        kernel_size=(1, 1),
                        padding=self.padding,
                        use_bias=False)(avg_pool1)
        bn1 = BatchNormalization()(conv14)
        act1 = Activation(self.activation)(bn1)
        act1 = Dropout(0.5)(act1)

        img_resize_b1 = Lambda(lambda x: K.resize_images(x,
                                                  height_factor=strides_b1[0],
                                                  width_factor=strides_b1[1],
                                                  data_format='channels_last',
                                                  interpolation='bilinear'
                                                  ))(act1)
        pool_output.append(img_resize_b1)
        # Pyramid Block 2
        height_b2 = K.int_shape(level5)[1]
        width_b2 = K.int_shape(level5)[2]

        pool_size_b2 = [int(np.round(float(height_b2) / pool_factor[1])),
                        int(np.round(float(width_b2) / pool_factor[1]))
                        ]
        strides_b2 = pool_size_b2
        avg_pool2 = AveragePooling2D(pool_size_b2,
                                     strides=strides_b2,
                                     padding=self.padding)(level5)
        conv15 = Conv2D(f[3],
                        kernel_size=(1, 1),
                        padding=self.padding,
                        use_bias=False)(avg_pool2)
        bn2 = BatchNormalization()(conv15)
        act2 = Activation(self.activation)(bn2)
        act2 = Dropout(0.5)(act2)

        img_resize_b2 = Lambda(lambda x: K.resize_images(x,
                                                  height_factor=strides_b2[0],
                                                  width_factor=strides_b2[1],
                                                  data_format='channels_last',
                                                  interpolation='bilinear'
                                                  ))(act2)
        pool_output.append(img_resize_b2)
        # Pyramid Block 3
        height_b3 = K.int_shape(level5)[1]
        width_b3 = K.int_shape(level5)[2]

        pool_size_b3 = [int(np.round(float(height_b3) / pool_factor[2])),
                        int(np.round(float(width_b3) / pool_factor[2]))
                        ]
        strides_b3 = pool_size_b3
        avg_pool3 = AveragePooling2D(pool_size_b3,
                                     strides=strides_b3,
                                     padding=self.padding)(level5)
        conv16 = Conv2D(f[3],
                        kernel_size=(1, 1),
                        padding=self.padding,
                        use_bias=False)(avg_pool3)
        bn3 = BatchNormalization()(conv16)
        act3 = Activation(self.activation)(bn3)
        act3 = Dropout(0.5)(act3)

        img_resize_b3 = Lambda(lambda x: K.resize_images(x,
                                                  height_factor=strides_b3[0],
                                                  width_factor=strides_b3[1],
                                                  data_format='channels_last',
                                                  interpolation='bilinear'
                                                  ))(act3)
        pool_output.append(img_resize_b3)
        # Pyramid Block 4
        height_b4 = K.int_shape(level5)[1]
        width_b4 = K.int_shape(level5)[2]

        pool_size_b4 = [int(np.round(float(height_b4) / pool_factor[3])),
                        int(np.round(float(width_b4) / pool_factor[3]))
                        ]
        strides_b4 = pool_size_b4
        avg_pool4 = AveragePooling2D(pool_size_b4,
                                     strides=strides_b4,
                                     padding=self.padding)(level5)
        conv17 = Conv2D(f[3],
                        kernel_size=(1, 1),
                        padding=self.padding,
                        use_bias=False)(avg_pool4)
        bn4 = BatchNormalization()(conv17)
        act4 = Activation(self.activation)(bn4)
        act4 = Dropout(0.5)(act4)

        img_resize_b4 = Lambda(lambda x: K.resize_images(x,
                                                  height_factor=strides_b4[0],
                                                  width_factor=strides_b4[1],
                                                  data_format='channels_last',
                                                  interpolation='bilinear'
                                                  ))(act4)
        pool_output.append(img_resize_b4)

        # Concatenate the pyramids
        concat1 = Concatenate(axis=-1)(pool_output)

        conv18 = Conv2D(f[3],
                        kernel_size=(1, 1),
                        use_bias=False)(concat1)
        print('Done until here')
        bn5 = BatchNormalization()(conv18)
        act5 = Activation(self.activation)(bn5)
        act5 = Dropout(0.5)(act5)

        conv19 = Conv2D(self.n_classes,
                        kernel_size=(3, 3),
                        padding=self.padding)(act5)
        # resizing
        img_resize_b4 = Lambda(lambda x: K.resize_images(x,
                                                  height_factor=32,
                                                  width_factor=32,
                                                  data_format='channels_last',
                                                  interpolation='bilinear'
                                                  ))(conv19)
        output_shape_f = Model(inputs, img_resize_b4).output_shape
        input_shape_f = Model(inputs, img_resize_b4).input_shape

        op_height_f = output_shape_f[1]
        op_width_f = output_shape_f[2]

        outputs = Reshape((op_height_f*op_width_f, -1))(img_resize_b4)

        outputs = Activation('softmax')(img_resize_b4)

        model = Model(inputs, outputs)

        if self.pre_trained is True:
          model = load_model('../trained_models/pspnet_inria.h5')

        return model
