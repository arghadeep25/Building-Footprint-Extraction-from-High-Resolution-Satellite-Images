"""
Name: Pyramid Scene Parsing Network (PSPNet) Model
Author: Arghadeep Mazumder
Version: 0.1
Description: PSPNet Architecture
"""

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add, concatenate
from keras import backend as K
import tensorflow as tf


class PSPNet():
    """ Pyramid Scene Parsing Network Model

        Parameters: - Image Size (default = 256)
                    - Kernel Size (default = (3, 3))
                    - Padding (default = 'same')
                    - Activation (default = 'relu')
                    - Pool Size (default = 2)
                    - Strides (default = 1)
                    - Max Pool Strides (default = 2)
                    - Up Sample (default = 2)
    """
    def __init__(self, image_size=256,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 pool_size=2,
                 strides=1,
                 max_pool_strides=2,
                 up_sample=2):
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.pool_size = pool_size
        self.strides = strides
        self.max_pool_strides = max_pool_strides
        self.up_sample = up_sample

    def network(self):
        # filters
        f = [64, 128, 256, 512, 1024, 2048]
        # inputs
        inputs = Input((self.image_size, self.image_size, 3))

        conv1 = Conv2D(f[0],
                       kernel_size=self.kernel_size,
                       padding=self.padding,
                       strides=2)(inputs)
        bn1 = BatchNormalization()(conv1)
        act1 = Activation(self.activation)(bn1)

        conv2 = Conv2D(f[0],
                       kernel_size=self.kernel_size,
                       padding=self.padding,
                       strides=1)(act1)
        bn2 = BatchNormalization()(conv2)
        act2 = Activation(self.activation)(bn2)

        conv3 = Conv2D(f[1],
                       kernel_size=self.kernel_size,
                       padding=self.padding,
                       strides=1)(act2)
        bn3 = BatchNormalization()(conv3)
        act3 = Activation(self.activation)(bn3)

        maxpool1 = MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding=self.padding)(act3)

        # comv block1 64 64 256 1 1 1
        conv4 = Conv2D(f[0],
                       kernel_size=1,
                       dilation_rate=1,
                       )(maxpool1)
        bn4 = BatchNormalization()(conv4)
        act4 = Activation(self.activation)(bn4)

        conv5 = Conv2D(f[0],
                       kernel_size=3,
                       strides=1,
                       padding=self.padding,
                       dilation_rate=1)(act4)
        bn5 = BatchNormalization()(conv5)
        act5 = Activation(self.activation)(bn5)

        conv6 = Conv2D(f[2],
                       kernel_size=1,
                       dilation_rate=1)(act5)
        bn6 = BatchNormalization()(conv6)

        conv7 = Conv2D(f[2],
                       kernel_size=1,
                       strides=1)(maxpool1)
        bn7 = BatchNormalization()(conv7)

        sum1 = Add()([bn6, bn7])
        sum1 = BatchNormalization()(sum1)

        #  indentity block 1
        # ip = sum1, d = 1 1 1 f = 64 64 256
        conv8 = Conv2D(f[0],
                       kernel_size=1,
                       dilation_rate=1)(sum1)
        bn8 = BatchNormalization()(conv8)
        act8 = Activation(self.activation)(bn8)

        conv9 = Conv2D(f[0],
                       kernel_size=3,
                       padding=self.padding,
                       dilation_rate=1)(act8)
        bn9 = BatchNormalization()(conv9)
        act9 = Activation(self.activation)(bn9)

        conv10 = Conv2D(f[2],
                        kernel_size=1,
                        dilation_rate=1)(act9)
        bn10 = BatchNormalization()(conv10)

        sum2 = Add()([bn10, sum1])
        sum2 = Activation(self.activation)(sum2)

        #  indentity block 2
        # ip = sum2, d = 1 1 1 f = 64 64 256
        conv11 = Conv2D(f[0],
                        kernel_size=1,
                        dilation_rate=1)(sum2)
        bn11 = BatchNormalization()(conv11)
        act11 = Activation(self.activation)(bn11)

        conv12 = Conv2D(f[0],
                        kernel_size=3,
                        padding=self.padding,
                        dilation_rate=1)(act11)
        bn12 = BatchNormalization()(conv12)
        act12 = Activation(self.activation)(bn12)

        conv13 = Conv2D(f[2],
                        kernel_size=1,
                        dilation_rate=1)(act12)
        bn13 = BatchNormalization()(conv13)

        sum3 = Add()([bn13, sum2])
        sum3 = Activation(self.activation)(sum3)

        # comv block2 128 128 512  s 2 d 1 1 1
        conv14 = Conv2D(f[1],
                        kernel_size=1,
                        dilation_rate=1,
                        )(sum3)
        bn14 = BatchNormalization()(conv14)
        act14 = Activation(self.activation)(bn14)

        conv15 = Conv2D(f[1],
                        kernel_size=3,
                        strides=2,
                        padding=self.padding,
                        dilation_rate=1)(act14)
        bn15 = BatchNormalization()(conv15)
        act15 = Activation(self.activation)(bn15)

        conv16 = Conv2D(f[3],
                        kernel_size=1,
                        dilation_rate=1)(act15)
        bn16 = BatchNormalization()(conv16)

        conv17 = Conv2D(f[3],
                        kernel_size=1,
                        strides=2)(sum3)
        bn17 = BatchNormalization()(conv17)

        sum4 = Add()([bn16, bn17])
        sum4 = BatchNormalization()(sum4)

        #  indentity block 3
        # ip = sum2, d = 1 1 1 f = 64 64 256
        conv18 = Conv2D(f[1],
                        kernel_size=1,
                        dilation_rate=1)(sum4)
        bn18 = BatchNormalization()(conv18)
        act18 = Activation(self.activation)(bn18)

        conv19 = Conv2D(f[1],
                        kernel_size=3,
                        padding=self.padding,
                        dilation_rate=1)(act18)
        bn19 = BatchNormalization()(conv19)
        act19 = Activation(self.activation)(bn19)

        conv20 = Conv2D(f[3],
                        kernel_size=1,
                        dilation_rate=1)(act19)
        bn20 = BatchNormalization()(conv20)

        sum5 = Add()([bn20, sum4])
        sum5 = Activation(self.activation)(sum5)

        #  indentity block 4
        # ip = sum5, d = 1 1 1 f = 128 128 512
        conv21 = Conv2D(f[1],
                        kernel_size=1,
                        dilation_rate=1)(sum5)
        bn21 = BatchNormalization()(conv21)
        act21 = Activation(self.activation)(bn21)

        conv22 = Conv2D(f[1],
                        kernel_size=3,
                        padding=self.padding,
                        dilation_rate=1)(act21)
        bn22 = BatchNormalization()(conv22)
        act22 = Activation(self.activation)(bn22)

        conv23 = Conv2D(f[3],
                        kernel_size=1,
                        dilation_rate=1)(act22)
        bn23 = BatchNormalization()(conv23)

        sum6 = Add()([bn23, sum5])
        sum6 = Activation(self.activation)(sum5)

        #  indentity block 5
        # ip = sum5, d = 1 1 1 f = 128 128 512
        conv24 = Conv2D(f[1],
                        kernel_size=1,
                        dilation_rate=1)(sum6)
        bn24 = BatchNormalization()(conv24)
        act24 = Activation(self.activation)(bn24)

        conv25 = Conv2D(f[1],
                        kernel_size=3,
                        padding=self.padding,
                        dilation_rate=1)(act24)
        bn25 = BatchNormalization()(conv25)
        act25 = Activation(self.activation)(bn25)

        conv26 = Conv2D(f[3],
                        kernel_size=1,
                        dilation_rate=1)(act25)
        bn26 = BatchNormalization()(conv26)

        sum7 = Add()([bn26, sum6])
        sum7 = Activation(self.activation)(sum7)

        # comv block3 256 256 1024  s 2 d 1 2 1
        conv27 = Conv2D(f[2],
                        kernel_size=1,
                        dilation_rate=1,
                        )(sum7)
        bn27 = BatchNormalization()(conv27)
        act27 = Activation(self.activation)(bn27)

        conv28 = Conv2D(f[2],
                        kernel_size=3,
                        strides=1,
                        padding=self.padding,
                        dilation_rate=2)(act27)
        bn28 = BatchNormalization()(conv28)
        act28 = Activation(self.activation)(bn28)

        conv29 = Conv2D(f[4],
                        kernel_size=1,
                        dilation_rate=1)(act28)
        bn29 = BatchNormalization()(conv29)

        conv30 = Conv2D(f[4],
                        kernel_size=1,
                        strides=1)(sum7)
        bn30 = BatchNormalization()(conv30)

        sum8 = Add()([bn29, bn30])
        sum8 = BatchNormalization()(sum8)

        #  indentity block 6
        # ip = sum5, d = 1 1 1 f = 128 128 512
        conv31 = Conv2D(f[2],
                        kernel_size=1,
                        dilation_rate=1)(sum8)
        bn31 = BatchNormalization()(conv31)
        act31 = Activation(self.activation)(bn31)

        conv32 = Conv2D(f[2],
                        kernel_size=3,
                        padding=self.padding,
                        dilation_rate=2)(act31)
        bn32 = BatchNormalization()(conv32)
        act32 = Activation(self.activation)(bn32)

        conv33 = Conv2D(f[4],
                        kernel_size=1,
                        dilation_rate=1)(act32)
        bn33 = BatchNormalization()(conv33)

        sum9 = Add()([bn33, sum8])
        sum9 = Activation(self.activation)(sum9)

        #  indentity block 7
        # ip = sum5, d = 1 1 1 f = 128 128 512
        conv34 = Conv2D(f[2],
                        kernel_size=1,
                        dilation_rate=1)(sum9)
        bn34 = BatchNormalization()(conv34)
        act34 = Activation(self.activation)(bn34)

        conv35 = Conv2D(f[2],
                        kernel_size=3,
                        padding=self.padding,
                        dilation_rate=2)(act34)
        bn35 = BatchNormalization()(conv35)
        act35 = Activation(self.activation)(bn35)

        conv36 = Conv2D(f[4],
                        kernel_size=1,
                        dilation_rate=1)(act35)
        bn36 = BatchNormalization()(conv36)

        sum10 = Add()([bn36, sum9])
        sum10 = Activation(self.activation)(sum10)

        #  indentity block 8
        # ip = sum5, d = 1 1 1 f = 128 128 512
        conv37 = Conv2D(f[2],
                        kernel_size=1,
                        dilation_rate=1)(sum10)
        bn37 = BatchNormalization()(conv37)
        act37 = Activation(self.activation)(bn37)

        conv38 = Conv2D(f[2],
                        kernel_size=3,
                        padding=self.padding,
                        dilation_rate=2)(act37)
        bn38 = BatchNormalization()(conv38)
        act38 = Activation(self.activation)(bn38)

        conv39 = Conv2D(f[4],
                        kernel_size=1,
                        dilation_rate=1)(act38)
        bn39 = BatchNormalization()(conv39)

        sum11 = Add()([bn39, sum10])
        sum11 = Activation(self.activation)(sum11)

        #  indentity block 9
        # ip = sum5, d = 1 1 1 f = 128 128 512
        conv40 = Conv2D(f[2],
                        kernel_size=1,
                        dilation_rate=1)(sum11)
        bn40 = BatchNormalization()(conv40)
        act40 = Activation(self.activation)(bn40)

        conv41 = Conv2D(f[2],
                        kernel_size=3,
                        padding=self.padding,
                        dilation_rate=2)(act40)
        bn41 = BatchNormalization()(conv41)
        act41 = Activation(self.activation)(bn41)

        conv42 = Conv2D(f[4],
                        kernel_size=1,
                        dilation_rate=1)(act41)
        bn42 = BatchNormalization()(conv42)

        sum12 = Add()([bn42, sum11])
        sum12 = Activation(self.activation)(sum12)

        #  indentity block 10
        # ip = sum5, d = 1 1 1 f = 128 128 512
        conv43 = Conv2D(f[2],
                        kernel_size=1,
                        dilation_rate=1)(sum12)
        bn43 = BatchNormalization()(conv43)
        act43 = Activation(self.activation)(bn43)

        conv44 = Conv2D(f[2],
                        kernel_size=3,
                        padding=self.padding,
                        dilation_rate=2)(act43)
        bn44 = BatchNormalization()(conv44)
        act44 = Activation(self.activation)(bn44)

        conv45 = Conv2D(f[4],
                        kernel_size=1,
                        dilation_rate=1)(act44)
        bn45 = BatchNormalization()(conv45)

        sum13 = Add()([bn45, sum12])
        sum13 = Activation(self.activation)(sum13)

        # comv block4 512 512 2048  s 1 d 1 4 1
        conv46 = Conv2D(f[3],
                        kernel_size=1,
                        dilation_rate=1,
                        )(sum13)
        bn46 = BatchNormalization()(conv46)
        act46 = Activation(self.activation)(bn46)

        conv47 = Conv2D(f[3],
                        kernel_size=3,
                        strides=1,
                        padding=self.padding,
                        dilation_rate=4)(act46)
        bn47 = BatchNormalization()(conv47)
        act47 = Activation(self.activation)(bn47)

        conv48 = Conv2D(f[5],
                        kernel_size=1,
                        dilation_rate=1)(act47)
        bn48 = BatchNormalization()(conv48)

        conv49 = Conv2D(f[5],
                        kernel_size=1,
                        strides=1)(sum13)
        bn49 = BatchNormalization()(conv49)

        sum14 = Add()([bn48, bn49])
        sum14 = BatchNormalization()(sum14)

        #  indentity block 11
        # ip = sum5, d = 1 1 1 f = 512 512 2048
        conv50 = Conv2D(f[3],
                        kernel_size=1,
                        dilation_rate=1)(sum14)
        bn50 = BatchNormalization()(conv50)
        act50 = Activation(self.activation)(bn50)

        conv51 = Conv2D(f[3],
                        kernel_size=3,
                        padding=self.padding,
                        dilation_rate=4)(act50)
        bn51 = BatchNormalization()(conv51)
        act51 = Activation(self.activation)(bn51)

        conv52 = Conv2D(f[5],
                        kernel_size=1,
                        dilation_rate=1)(act51)
        bn52 = BatchNormalization()(conv52)

        sum15 = Add()([bn52, sum14])
        sum15 = Activation(self.activation)(sum15)

        #  indentity block 12
        # ip = sum5, d = 1 1 1 f = 512 512 2048
        conv53 = Conv2D(f[3],
                        kernel_size=1,
                        dilation_rate=1)(sum15)
        bn53 = BatchNormalization()(conv53)
        act53 = Activation(self.activation)(bn53)

        conv54 = Conv2D(f[3],
                        kernel_size=3,
                        padding=self.padding,
                        dilation_rate=4)(act53)
        bn54 = BatchNormalization()(conv54)
        act54 = Activation(self.activation)(bn54)

        conv55 = Conv2D(f[5],
                        kernel_size=1,
                        dilation_rate=1)(act54)
        bn55 = BatchNormalization()(conv55)

        sum16 = Add()([bn55, sum15])
        sum16 = Activation(self.activation)(sum16)

        #  pyramid pooling block
        concat1 = [sum16]
        bins = [1, 2, 3, 6]

        height = sum16.shape[1].value
        width = sum16.shape[2].value

        for bin in bins:
            x = AveragePooling2D(pool_size=(height/bin, width/bin),
                                 strides=(height/bin, width/bin))(sum16)
            x = Conv2D(f[3], kernel_size=1)(x)
            x = Lambda(lambda x: tf.image.resize_images(x,(height, width)))(x)
            concat1.append(x)

        concat2 = concatenate(concat1)

        conv57 = Conv2D(f[3],
                        kernel_size=1,
                        padding=self.padding)(concat2)
        bn57 = BatchNormalization()(conv57)
        act57 = Activation(self.activation)(bn57)
        drop = Dropout(0.1)(act57)

        conv58 = Conv2D(1, kernel_size=1)(drop)

        # Final Layer
        outputs = Conv2DTranspose(1,
                                  kernel_size=16,
                                  strides=8,
                                  padding=self.padding,
                                  activation='softmax')(conv58)

        model = Model(inputs, outputs)
        return model
