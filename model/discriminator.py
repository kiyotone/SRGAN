import torch
import torch.nn as nn
from .blocks import ConvBlock

class Discriminator(nn.Module):
    def __init__(self,in_channels=3, features= [64,64,128,128,256,256,512,512]):
        super(Discriminator, self).__init__()
        blocks = []
        for idx , feature in enumerate(features):
            blocks.append(ConvBlock(in_channels,feature,kernel_size=3,stride=1,padding=1,discriminator=True , use_act= True, use_batchnorm=False if idx == 0 else True))
            in_channels = feature
        self.blocks = nn.Sequential(*blocks)   
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6,6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,1),
        
        )
        
    def forward(self,x):
        x = self.blocks(x)
        return self.classifier(x)
    

def test():
    x = torch.randn((5, 3, 24, 24))  # Example input size (batch_size=5, channels=3, height=256, width=256)
    model = Discriminator()
    preds = model(x)
    print(preds.shape)  # Expected output: torch.Size([5, 1])        
        
if __name__ == "__main__":
    test()
