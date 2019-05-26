import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, input_channel, 3, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class InConv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(InConv,self).__init__()
        self.conv = DoubleConv(input_channel, output_channel)

    def forward(self, x):
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2),
                                    DoubleConv(input_channel, output_channel)
                                    )
    def forward(self, x):
        x = self.mpconv(x)
        return x

class Up(nn.Module):
    def __init__(self, input_channel, output_channel, bilinear = True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                mode='bilinear',
                                align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(input_channel//2,
                                        input_channel//2,
                                        stride=2)
        self.conv = DoubleConv(input_channel, output_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim = 1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
