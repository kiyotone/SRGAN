import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added {project_root} to sys.path")


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
import random
from torchvision import transforms
from model.generator import Generator
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, LR_path, HR_path, scale=4, patch_size=32, in_memory=True, transform=None):
        self.LR_path = LR_path
        self.HR_path = HR_path
        self.scale = scale
        self.patch_size = patch_size
        self.transform = transform
        self.in_memory = in_memory

        self.LR_img_names = os.listdir(LR_path)
        self.HR_img_names = os.listdir(HR_path)

        # Ensure images are correctly paired
        self.LR_img_names.sort()
        self.HR_img_names.sort()

        if in_memory:
            self.LR_images = []
            self.HR_images = []
            for img_name in self.LR_img_names:
                lr_img_path = os.path.join(self.LR_path, img_name)
                hr_img_path = os.path.join(self.HR_path, img_name)

                try:
                    # Load the images and check if they are valid
                    lr_img = cv2.imread(lr_img_path)
                    hr_img = cv2.imread(hr_img_path)
                    
                    # Check if the image has valid dimensions
                    if lr_img.shape[0] == 0 or lr_img.shape[1] == 0 or hr_img.shape[0] == 0 or hr_img.shape[1] == 0:
                        print(f"Skipping corrupted image: {img_name}")
                        continue

                    self.LR_images.append(lr_img)
                    self.HR_images.append(hr_img)

                except Exception as e:
                    print(f"Error loading image {img_name}: {e}")
                    continue

    def __len__(self):
        if self.in_memory:
            return len(self.LR_images)
        return len(self.LR_img_names)

    def __getitem__(self, idx):
        if self.in_memory:
            LR_img = self.LR_images[idx]
            HR_img = self.HR_images[idx]
        else:
            LR_img = cv2.imread(os.path.join(self.LR_path, self.LR_img_names[idx]))
            HR_img = cv2.imread(os.path.join(self.HR_path, self.HR_img_names[idx]))

            if LR_img is None or HR_img is None:
                raise ValueError(f"One or both images are not loaded correctly for index {idx}.")

        LR_img = LR_img.astype(np.float32) / 255
        HR_img = HR_img.astype(np.float32) / 255

        sample = {'LR': LR_img, 'HR': HR_img}

        if self.transform:
            sample = self.transform(sample)

        sample['LR'] = torch.from_numpy(sample['LR']).permute(2, 0, 1)
        sample['HR'] = torch.from_numpy(sample['HR']).permute(2, 0, 1)

        return sample

class Crop(object):
    def __init__(self, scale, patch_size):
        self.scale = scale
        self.patch_size = patch_size

    def __call__(self, sample):
        LR_img, HR_img = sample['LR'], sample['HR']

        h, w, _ = LR_img.shape
        hr_h, hr_w, _ = HR_img.shape

        # Ensure patch size is valid
        if h < self.patch_size or w < self.patch_size or hr_h < self.patch_size * self.scale or hr_w < self.patch_size * self.scale:
            # print("Invalid patch size. Returning original images.")
            return sample

        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)

        LR_img = LR_img[y:y+self.patch_size, x:x+self.patch_size, :]
        HR_img = HR_img[y*self.scale:y*self.scale+self.patch_size*self.scale, x*self.scale:x*self.scale+self.patch_size*self.scale, :]

        # Validate non-empty images after cropping
        if LR_img.size == 0 or HR_img.size == 0:
            # print("Empty image detected after cropping. Returning original images.")
            return sample

        return {'LR': LR_img, 'HR': HR_img}

class Augmentation(object):
    def __call__(self, sample):
        LR_img, HR_img = sample['LR'], sample['HR']

        if random.random() > 0.5:
            LR_img = np.fliplr(LR_img).copy()
            HR_img = np.fliplr(HR_img).copy()

        if random.random() > 0.5:
            LR_img = np.flipud(LR_img).copy()
            HR_img = np.flipud(HR_img).copy()

        if random.random() > 0.5:
            LR_img = LR_img.transpose(1, 0, 2).copy()
            HR_img = HR_img.transpose(1, 0, 2).copy()

        # Validate non-empty images after augmentation
        if LR_img.size == 0 or HR_img.size == 0:
            # print("Empty image detected after augmentation. Returning original images.")
            return sample

        return {'LR': LR_img, 'HR': HR_img}

def collate_fn(batch):
    # Filter out invalid samples
    batch = [sample for sample in batch if sample['LR'].shape == sample['HR'].shape]
    # Remove NoneType samples
    batch = [sample for sample in batch if sample is not None]

    return torch.utils.data.dataloader.default_collate(batch)

def get_dataloader(LR_path= 'dataset/train/low_res', HR_path= 'dataset/train/high_res', patch_size=32 , scale=4 , batch_size=16):
    
    scale = 4
    patch_size = 32

    transform = transforms.Compose([Crop(scale, patch_size), Augmentation()])

    dataset = MyDataset(LR_path, HR_path, scale=scale, patch_size=patch_size, in_memory=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    return dataloader

if __name__ == "__main__":
    LR_path = 'dataset/check/low_res'
    HR_path = 'dataset/check/high_res'
    batch_size = 16
    patch_size = 32
    scale = 4

    transform = transforms.Compose([Crop(scale, patch_size), Augmentation()])

    dataset = MyDataset(LR_path, HR_path, scale=scale, patch_size=patch_size, in_memory=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    gen = Generator(3, 3)
            
    
    for batch in dataloader:
        LR = batch['LR']
        HR = batch['HR']
        break
    
    plt.imshow(LR[0].permute(1, 2, 0).numpy())
    plt.show()
    
    data = gen(LR).detach().numpy()[0].transpose(1, 2, 0)
    print(data.shape)
    # Imshow data

    plt.imshow(data)
    plt.show()
    
