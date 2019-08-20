from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Reshape
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling2D, UpSampling2D
from .layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


class SegNet:
    def __init__(self,image_size = 256,
                 kernel_size = (3, 3),
                 padding = 'same',
                 activation = 'relu',
                 pool_size = (2, 2),
                 strides = 1,
                 max_pool_strides = 2,
                 up_sample = 2):
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.pool_size = pool_size
        self.strides = strides
        self.max_pool_strides = max_pool_strides
        self.up_sample = up_sample

    def network(self):

        img_input = Input((self.image_size, self.image_size, 3))
        x = img_input
        # Encoder
        x = Conv2D(64, self.kernel_size, border_mode="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(128, self.kernel_size, border_mode="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(256, self.kernel_size, border_mode="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(512, self.kernel_size, border_mode="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Decoder
        x = Conv2D(512, self.kernel_size, border_mode="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(256, self.kernel_size, border_mode="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, self.kernel_size, border_mode="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, self.kernel_size, border_mode="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(2, 1, 1, border_mode="valid")(x)
        x = Reshape((self.image_size*self.image_size, 2))(x)
        x = Activation("softmax")(x)
        model = Model(img_input, x)

        return model
