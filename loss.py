import torch
import torch.nn as nn
import torchvision.models as models


class VGGLoss(nn.Module):
    """Perceptual loss based on VGG19 feature maps."""
    
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:36].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False  # Freeze VGG model
        self.vgg = vgg
        self.criterion = nn.MSELoss()  # Pixel-wise similarity loss

    def forward(self, fake, real):
        fake_features = self.vgg(fake)
        real_features = self.vgg(real)
        return self.criterion(fake_features, real_features)


class SRGANLosses:
    """Collection of losses for training SRGAN."""
    
    def __init__(self, device, vgg_weight=1e-3):
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()  # Adversarial loss
        self.vgg_loss = VGGLoss(device)
        self.vgg_weight = vgg_weight

    def generator_loss(self, fake_img, real_img, fake_pred):
        """Compute total generator loss."""
        content_loss = self.mse_loss(fake_img, real_img)
        adversarial_loss = self.bce_loss(fake_pred, torch.ones_like(fake_pred))
        return content_loss + self.vgg_weight * adversarial_loss

    def discriminator_loss(self, real_pred, fake_pred):
        """Compute discriminator loss."""
        real_loss = self.bce_loss(real_pred, torch.ones_like(real_pred))
        fake_loss = self.bce_loss(fake_pred, torch.zeros_like(fake_pred))
        return real_loss + fake_loss
