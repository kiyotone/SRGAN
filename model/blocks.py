import torch 
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, discriminator=False,use_act=True , use_batchnorm=True , **kwargs):
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_batchnorm)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.act = (nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(out_channels)) if use_act else nn.Identity()
        

    def forward(self, x):
        return self.act(self.bn(self.conv(x))) if self.use_act else self.bn(self.conv(x))
    

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, 3, 1 ,1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(in_channels)
    
    def forward(self, x):
        return self.act(self.pixel_shuffle(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x

