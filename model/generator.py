import torch
import torch.nn as nn
from .blocks import ConvBlock, ResidualBlock, UpSampleBlock

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsamples = nn.Sequential(UpSampleBlock(num_channels, 2), UpSampleBlock(num_channels, 2))
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        return torch.tanh(self.final(x))


def test():
    x = torch.randn((5, 3, 64, 64))  # Example input size (batch_size=5, channels=3, height=64, width=64)
    model = Generator(3, 3)
    preds = model(x)
    print(preds.shape)  # Should print torch.Size([5, 3, 256, 256]) due to upsampling

if __name__ == "__main__":
    test()
