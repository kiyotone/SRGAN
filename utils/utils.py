import torch
import os
from PIL import Image
from torchvision.utils import save_image
import config
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import random
from random import choice


def gradient_penalty(critic , real , fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)
    interpolated_images.requires_grad_(True)
    critic_interpolates = critic(interpolated_images)
    gradients = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=critic_interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(BATCH_SIZE, -1)
    gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
    return gradient_penalty

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print(f"=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print(f"=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def plot_examples(low_res_folder, gen):
    files = os.listdir(low_res_folder)
    file = np.random.choice(files)
    
    gen.eval()
    
    image = Image.open(os.path.join(low_res_folder, file)).convert("RGB")  # Convert image to RGB
    with torch.no_grad():
        input_tensor = config.test_transform(image=np.asarray(image))["image"].unsqueeze(0).to(config.DEVICE)
        upscaled_img = gen(input_tensor)
    
    # Normalize the output to [0, 1] range
    upscaled_img = (upscaled_img * 0.5) + 0.5
    
    
    # Save the images
    image_tensor = config.test_transform(image=np.asarray(image))["image"]
    index = os.listdir("saved")
    idx = len(index)
    
    save_image(image_tensor, f"saved/{idx}.png")
    save_image(upscaled_img, f"saved/{idx + 1}.png")
    
    gen.train()