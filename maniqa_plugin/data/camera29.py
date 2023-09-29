from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
import random
import json


class CameraCropDataset(Dataset):
    def __init__(self, gt_file, img_path, txt_file_name, quality_factor, transform=None):

        with open(gt_file, "r") as f:
            self.target = json.load(f)
        
        self.data_dir = img_path
        self.quality_factor = quality_factor
        self.transform = transform
        
        with open(txt_file_name, 'r') as f:
            self.data = f.readlines()
            
        self.quality_values = [item[self.quality_factor] for item in list(self.target.values())]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name, camera = self.data[idx].split()

        image = np.asarray(Image.open(os.path.join(self.data_dir, img_name)))
        gt_dict = self.target[camera]
        pred_target = float(gt_dict[self.quality_factor])
        pred_target = (pred_target - min(self.quality_values)) / (max(self.quality_values) - min(self.quality_values))

        if self.transform:
            image = self.transform(image)

        return {'image':image, 'target':pred_target}
