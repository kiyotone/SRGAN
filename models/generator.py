import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)  # Skip connection

class Generator(nn.Module):
    def __init__(self, upscale_factor=2, num_residual_blocks=4):
        super(Generator, self).__init__()
        
        # Initial convolution layer
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        
        # Second convolution layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Upsampling layers (pixel shuffle for super-resolution)
        upsample_layers = []
        for _ in range(int(torch.log2(torch.tensor(upscale_factor)))):
            upsample_layers.append(nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1))
            upsample_layers.append(nn.PixelShuffle(2))
            upsample_layers.append(nn.PReLU())
        self.upsample = nn.Sequential(*upsample_layers)
        
        # Final output layer
        self.final = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)
        
    def forward(self, x):
        initial = self.initial(x)
        res = self.residual_blocks(initial)
        conv2 = self.conv2(res) + initial  # Skip connection from initial conv layer
        upsampled = self.upsample(conv2)
        output = self.final(upsampled)
        return output
