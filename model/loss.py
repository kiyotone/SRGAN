import torch.nn as nn
from torchvision.models import vgg19
import config

# phi_5,4 5th conv layer before maxpooling but after activation

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        # Ensure input and target have the same size
        assert input.size() == target.size(), f"Input and target must have the same size. Got {input.size()} and {target.size()}."

        # Get VGG features
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)

        # Check the size of VGG features
        assert vgg_input_features.size() == vgg_target_features.size(), (
            f"VGG features for input and target must have the same size. "
            f"Got {vgg_input_features.size()} and {vgg_target_features.size()}."
        )

        return self.loss(vgg_input_features, vgg_target_features)
