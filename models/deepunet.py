"""
Name: Deep Residual U-Net Model
Author: Arghadeep Mazumder
Version: 0.1
Description: Deep Residual U-Net Architecture
"""
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Activation, Conv2D, \
                         UpSampling2D, Concatenate, Input, Add


class DeepUNet():
    """ Deep Residual U-Net Model

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
                 kernel_size=(3,3),
                 padding='same',
                 activation='relu',
                 pool_size=2,
                 strides=1,
                 max_pool_strides=2,
                 up_sample=2,
                 pre_trained=False):
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.pool_size = pool_size
        self.strides = strides
        self.max_pool_strides = max_pool_strides
        self.up_sample = up_sample
        self.pre_trained = pre_trained

    def network(self):
        # filters
        f = [16, 32, 64, 128, 256]
        # inputs
        inputs = Input((self.image_size, self.image_size, 3))

        # Encoder
        conv1 = Conv2D(f[0],
                       kernel_size=self.kernel_size,
                       padding=self.padding,
                       strides=self.strides)(inputs)
        bn1 = BatchNormalization()(conv1)
        act1 = Activation(self.activation)(bn1)

        conv2 = Conv2D(f[0],
                       kernel_size=(1,1),
                       padding=self.padding,
                       strides=self.strides)(inputs)
        bn2 = BatchNormalization()(conv2)

        sum1 = Add()([act1, bn2])

        # residaul block 1
        bn3 = BatchNormalization()(sum1)
        act3 = Activation(self.activation)(bn3)
        conv3 = Conv2D(f[1],
                       kernel_size=self.kernel_size,
                       padding=self.padding,
                       strides=2)(act3)
        bn4 = BatchNormalization()(conv3)
        act4 = Activation(self.activation)(bn4)
        conv4 = Conv2D(f[1],
                       kernel_size=self.kernel_size,
                       padding=self.padding,
                       strides=1)(act4)
        conv5 = Conv2D(f[1],
                       kernel_size=(1, 1),
                       padding=self.padding,
                       strides=2)(sum1)
        bn5 = BatchNormalization()(conv5)

        sum2 = Add()([bn5, conv4])

        # residual block 2
        bn6 = BatchNormalization()(sum2)
        act6 = Activation(self.activation)(bn6)
        conv6 = Conv2D(f[2],
                       kernel_size=self.kernel_size,
                       padding=self.padding,
                       strides=2)(act6)
        bn7 = BatchNormalization()(conv6)
        act7 = Activation(self.activation)(bn7)
        conv7 = Conv2D(f[2],
                       kernel_size=self.kernel_size,
                       padding=self.padding,
                       strides=1)(act7)
        conv8 = Conv2D(f[2],
                       kernel_size=(1, 1),
                       padding=self.padding,
                       strides=2)(sum2)
        bn8 = BatchNormalization()(conv8)

        sum3 = Add()([bn8, conv7])

        # residual block 3
        bn9 = BatchNormalization()(sum3)
        act9 = Activation(self.activation)(bn9)
        conv9 = Conv2D(f[3],
                       kernel_size=self.kernel_size,
                       padding=self.padding,
                       strides=2)(act9)
        bn10 = BatchNormalization()(conv9)
        act10 = Activation(self.activation)(bn10)
        conv10 = Conv2D(f[3],
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        strides=1)(act10)
        conv11 = Conv2D(f[3],
                        kernel_size=(1, 1),
                        padding=self.padding,
                        strides=2)(sum3)
        bn11 = BatchNormalization()(conv11)

        sum4 = Add()([bn11, conv10])

        # residual block 4
        bn12 = BatchNormalization()(sum4)
        act12 = Activation(self.activation)(bn12)
        conv12 = Conv2D(f[4],
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        strides=2)(act12)
        bn13 = BatchNormalization()(conv12)
        act13 = Activation(self.activation)(bn13)
        conv13 = Conv2D(f[4],
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        strides=1)(act13)
        conv14 = Conv2D(f[4],
                        kernel_size=(1, 1),
                        padding=self.padding,
                        strides=2)(sum4)
        bn14 = BatchNormalization()(conv14)

        sum5 = Add()([bn14, conv13])

        # Bridge
        bn15 = BatchNormalization()(sum5)
        act15 = Activation(self.activation)(bn15)
        conv15 = Conv2D(f[4],
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        strides=1)(act15)

        bn16 = BatchNormalization()(conv15)
        act16 = Activation(self.activation)(bn16)
        conv16 = Conv2D(f[4],
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        strides=1)(act16)

        # Decoder
        # decode block 1
        us1 = UpSampling2D((2, 2))(conv16)
        conc1 = Concatenate()([us1, sum4])

        bn17 = BatchNormalization()(conc1)
        act17 = Activation(self.activation)(bn17)
        conv17 = Conv2D(f[4],
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        strides=1)(act17)
        bn18 = BatchNormalization()(conv17)
        act18 = Activation(self.activation)(bn18)
        conv18 = Conv2D(f[4],
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        strides=1)(act18)
        conv19 = Conv2D(f[4],
                        kernel_size=(1, 1),
                        padding=self.padding,
                        strides=1)(sum4)
        bn19 = BatchNormalization()(conv19)

        sum6 = Add()([bn19, conv18])

        # decode block 2
        us2 = UpSampling2D((2, 2))(sum6)
        conc2 = Concatenate()([us2, sum3])

        bn20 = BatchNormalization()(conc2)
        act20 = Activation(self.activation)(bn20)
        conv20 = Conv2D(f[3],
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        strides=1)(act20)
        bn21 = BatchNormalization()(conv20)
        act21 = Activation(self.activation)(bn21)
        conv21 = Conv2D(f[3],
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        strides=1)(act21)
        conv22 = Conv2D(f[3],
                        kernel_size=(1, 1),
                        padding=self.padding,
                        strides=1)(sum3)
        bn22 = BatchNormalization()(conv22)

        sum7 = Add()([bn22, conv21])

        # decode block 3
        us3 = UpSampling2D((2, 2))(sum7)
        conc3 = Concatenate()([us3, sum2])

        bn23 = BatchNormalization()(conc3)
        act23 = Activation(self.activation)(bn23)
        conv23 = Conv2D(f[2],
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        strides=1)(act23)
        bn24 = BatchNormalization()(conv23)
        act24 = Activation(self.activation)(bn24)
        conv24 = Conv2D(f[2],
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        strides=1)(act24)
        conv25 = Conv2D(f[2],
                        kernel_size=(1, 1),
                        padding=self.padding,
                        strides=1)(sum2)
        bn25 = BatchNormalization()(conv25)

        sum8 = Add()([bn25, conv24])

        # decode block 4
        us4 = UpSampling2D((2, 2))(sum8)
        conc4 = Concatenate()([us4, sum1])

        bn26 = BatchNormalization()(conc4)
        act26 = Activation(self.activation)(bn26)
        conv26 = Conv2D(f[1],
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        strides=1)(act26)
        bn27 = BatchNormalization()(conv26)
        act27 = Activation(self.activation)(bn27)
        conv27 = Conv2D(f[1],
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        strides=1)(act27)
        conv28 = Conv2D(f[1],
                        kernel_size=(1, 1),
                        padding=self.padding,
                        strides=1)(sum1)
        bn28 = BatchNormalization()(conv28)

        sum9 = Add()([bn28, conv27])

        # Final Layer
        outputs = Conv2D(1,
                         kernel_size=(1, 1),
                         padding="same",
                         activation="sigmoid")(sum9)
        model = Model(inputs, outputs)

        if self.pre_trained is True:
            model = load_model('../trained_models/deepunet_inria.h5')
        return model
