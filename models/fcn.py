"""
Name: Fully Convolutional Network
Author: Arghadeep Mazumder
Version: 0.1
Description: FCN Architecture
"""

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf


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
    def __init__(self,num_classes = 1,
                 input_shape = (256,256,3),
                 lr_init = 0.0001,
                 lr_decay=0.0005):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.lr_init = lr_init
        self.lr_decay = lr_decay
#         self.vgg_weight_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        self.vgg_weight_path = None

    def Convblock(self,channel_dimension, block_no, no_of_convs) :
      Layers = []
      for i in range(no_of_convs) :

          Conv_name = "conv"+str(block_no)+"_"+str(i+1)

          # A constant kernel size of 3*3 is used for all convolutions
          Layers.append(Convolution2D(channel_dimension,kernel_size = (3,3),padding = "same",activation = "relu",name = Conv_name))

      Max_pooling_name = "pool"+str(block_no)

      #Addding max pooling layer
      Layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = Max_pooling_name))

      return Layers

    def FCN_8_helper(self,image_size=256):
      model = Sequential()
      model.add(Permute((1,2,3),input_shape = (image_size,image_size,3)))

      for l in self.Convblock(64,1,2) :
          model.add(l)

      for l in self.Convblock(128,2,2):
          model.add(l)

      for l in self.Convblock(256,3,3):
          model.add(l)

      for l in self.Convblock(512,4,3):
          model.add(l)

      for l in self.Convblock(512,5,3):
          model.add(l)

      model.add(Convolution2D(4096,kernel_size=(7,7),padding = "same",activation = "relu",name = "fc6"))

      #Replacing fully connnected layers of VGG Net using convolutions
      model.add(Convolution2D(4096,kernel_size=(1,1),padding = "same",activation = "relu",name = "fc7"))

      # Gives the classifications scores for each of the 21 classes including background
      model.add(Convolution2D(21,kernel_size=(1,1),padding="same",activation="relu",name = "score_fr"))

      Conv_size = model.layers[-1].output_shape[2] #16 if image size if 512
      #print(Conv_size)

      model.add(Deconvolution2D(21,kernel_size=(4,4),strides = (2,2),padding = "valid",activation=None,name = "score2"))

      # O = ((I-K+2*P)/Stride)+1
      # O = Output dimesnion after convolution
      # I = Input dimnesion
      # K = kernel Size
      # P = Padding

      # I = (O-1)*Stride + K
      Deconv_size = model.layers[-1].output_shape[2] #34 if image size is 512*512

      #print(Deconv_size)
      # 2 if image size is 512*512
      Extra = (Deconv_size - 2*Conv_size)

      #print(Extra)

      #Cropping to get correct size
      model.add(Cropping2D(cropping=((0,Extra),(0,Extra))))

      return model

    def network(self):
      fcn_8 = self.FCN_8_helper(256)
      #Calculating conv size after the sequential block
      #32 if image size is 512*512
      Conv_size = fcn_8.layers[-1].output_shape[2]

      #Conv to be applied on Pool4
      skip_con1 = Convolution2D(1,kernel_size=(1,1),padding = "same",activation=None, name = "score_pool4")

      #Addig skip connection which takes adds the output of Max pooling layer 4 to current layer
      Summed = add(inputs = [skip_con1(fcn_8.layers[14].output),fcn_8.layers[-1].output])

      #Upsampling output of first skip connection
      x = Deconvolution2D(1,kernel_size=(4,4),strides = (2,2),padding = "valid",activation=None,name = "score4")(Summed)
      x = Cropping2D(cropping=((0,2),(0,2)))(x)


      #Conv to be applied to pool3
      skip_con2 = Convolution2D(1,kernel_size=(1,1),padding = "same",activation=None, name = "score_pool3")

      #Adding skip connection which takes output og Max pooling layer 3 to current layer
      Summed = add(inputs = [skip_con2(fcn_8.layers[10].output),x])

      #Final Up convolution which restores the original image size
      Up = Deconvolution2D(1,kernel_size=(16,16),strides = (8,8),
                           padding = "valid",activation = None,name = "upsample")(Summed)

      #Cropping the extra part obtained due to transpose convolution
      final = Cropping2D(cropping = ((0,8),(0,8)))(Up)


      return Model(fcn_8.input, final)
