# https://github.com/alishdipani/U-net-Pytorch/blob/master/networks.py
import torch
import torch.nn as nn
from torch.autograd import Variable

class unet(nn.Module):

    """Modelling the U-Net architecture

        - Convolution 1: (Input Dimension = 256x256x3
                          Output Dimension = 128x128x16)

        - Convolution 2: (Input Dimension = 128x128x16
                          Output Dimension = 64x64x32)

        - Convolution 3: (Input Dimension = 64x64x32
                           Output Dimension = 64x64x64)

        - De-Convolution 1: (Input Dimension = 64x64x96
                            Output Dimension = 128x128x32)

        - De-Convolution 2: (Input Dimension = 128x128x48
                             Output Dimension = 256x256x16)

        - De-Convolution 3: (Input Dimension = 256x256x19
                             Output Dimnesion = 256x256x1)

    """

    def __init__(self):
        super(unet, self).__init__()

        # Convolution 1
        # Input Tensor Dimension = 256x256x3
        self.conv1 = nn.Conv2d(in_channels = 3,
                            out_channels = 16,
                            kernel_size = 5,
                            stride = 1,
                            padding = 2)
        #  Xavier Initialization
        nn.init.xavier_uniform(self.conv1.weight)
        self.activ_1 = nn.ELU()
        # Pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size = 2,
                                return_indices = True)
        # Output Tensor Dimension = 128x128x16

        # Convolution 2
        # Input Tensor Dimension = 128x128x16
        self.conv2 = nn.Conv2d(in_channels = 16,
                            out_channels = 32,
                            kernel_size = 3,
                            padding = 1)
        nn.init.xavier_uniform(self.conv2.weight)
        self.activ_2 = nn.ELU()
        # Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size = 2,
                                return_indices = True)
        # Output Tensor Dimension = 64x64x32

        # Convolution 3
        # Input Tensor Dimension = 64x64x32
        self.conv3 = nn.Conv2d(in_channels = 32,
                            out_channels = 64,
                            kernel_size = 3,
                            padding=1)
        nn.init.xavier_uniform(self.conv3.weight)
        self.activ_3 = nn.ELU()
        # Output Tensor Dimension = 64x64x64

        #  32 channel output of pool2 is concatenated

        # De-Convolution 1
        #  Input Tensor Dimension = 64x64x96
        self.deconv1 = nn.ConvTranspose2d(in_channels = 96,
                                        out_channels = 32,
                                        kernel_size = 3,
                                        padding = 1)
        nn.init.xavier_uniform(self.deconv1.weight)
        self.activ_4 = nn.ELU()
        # UnPooling 1
        self.unpool1 = nn.MaxUnpool2d(kernel_size = 2)
        # Output Tensor Dimension = 128x128x32

        # 16 channel output of pool1 is concatenated

        # De-Convolution 2
        # Input Tensor Dimension = 128x128x48
        self.deconv2 = nn.ConvTranspose2d(in_channels = 48,
                                        out_channels = 16,
                                        kernel_size = 3,
                                        padding=1)
        nn.init.xavier_uniform(self.deconv2.weight)
        self.activ_5 = nn.ELU()
        # UnPooling 2
        self.unpool2 = nn.MaxUnpool2d(kernel_size = 2)
        #  Output Tensor Dimension = 256x256x16

        # 3 channel input is concatenated

        # De-Convolution 3
        # Input Tensor Dimension = 256x256x19
        self.deconv3 = nn.ConvTranspose2d(in_channels =19,
                                        out_channels = 1,
                                        kernel_size = 5,
                                        padding = 2)
        nn.init.xavier_uniform(self.deconv3.weight)

        self.activ_6 = nn.Sigmoid()
        # Output Tensor Dimension = 256x256x1

    def forward(self,x):

        out_1 = x
        out = self.conv1(x)
        out = self.activ_1(out)
        size1 = out.size()
        out, indices1 = self.pool1(out)
        out_2 = out
        out = self.conv2(out)
        out = self.activ_2(out)
        size2 = out.size()
        out, indices2 = self.pool2(out)
        out_3 = out
        out = self.conv3(out)
        out = self.activ_3(out)

        out = torch.cat((out, out, 3), dim = 1)

        out = self.deconv1(out)
        out = self.activ_4(out)
        out = self.unpool1(out, indices2, size2)

        out = torch.cat((out, out, 2), dim = 1)

        out = self.deconv2(out)
        out = self.activ_5(out)
        out = self.unpool2(out, indices1, size1)

        out = torch.cat((out, out, 1), dim = 1)

        out = self.deconv3(out)
        out = self.activ_6(out)
        out = out
        return out
