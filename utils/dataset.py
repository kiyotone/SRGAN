from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import config

class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)
        
        for label, class_name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, class_name))[:30]
            self.data += list(zip(files, [label]*len(files)))
            
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])
        
        img = Image.open(os.path.join(root_and_dir, img_file)).convert("RGB")  # Convert to RGB
        img = np.array(img)
        
        # Apply both transforms
        img = config.both_transforms(image=img)["image"]
        
        # Apply high-res and low-res transforms
        high_res = config.highres_transforms(image=img)["image"]
        low_res = config.lowres_transforms(image=img)["image"]
        
        return high_res, low_res

def test():
    dataset = MyImageFolder(config.TRAIN_DIR)
    loader = DataLoader(dataset, batch_size=1, num_workers=config.NUM_WORKERS)
    
    for high_res, low_res in loader:
        print(high_res.shape, low_res.shape)
        break

if __name__ == "__main__":
    test()
