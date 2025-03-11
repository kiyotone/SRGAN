import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from utils.image_utils import ImageUtils

class SRGANDataSet(Dataset):
    def __init__(self, image_dir, transform=None, device='cpu'):
        self.image_dir = image_dir
        self.transform = transform
        self.device = device  # Store device (e.g., 'cuda' or 'cpu')

        self.image_list = os.listdir(os.path.join(self.image_dir, 'high_res'))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'high_res', self.image_list[idx])).convert('RGB')
        hr_image = ImageUtils.image_to_tensor(image)  # Converts to tensor
        lr_image = ImageUtils.downsample(hr_image, 2)

        # Ensure correct shape (remove singleton dimension if exists)
        hr_image = hr_image.squeeze(0)  # Remove extra dimension if present
        lr_image = lr_image.squeeze(0)  # Same for low-res image

        return lr_image, hr_image


    def show_image(self, idx):
        hr_image, lr_image = self[idx]  # Calls __getitem__(idx)
        ImageUtils.display_images([hr_image.cpu(), lr_image.cpu()], ['High Resolution', 'Low Resolution'])
