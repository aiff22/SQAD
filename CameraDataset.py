from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
import random
import json

with open('camera_ground_truth.json', "r") as f:
    target = json.load(f)

class CameraCropDataset(Dataset):
    def __init__(self, label_file, data_dir, quality_factor, transform=None):

        self.label_file = label_file
        self.data_dir = data_dir
        self.quality_factor = quality_factor
        self.transform = transform
        
        with open(self.label_file, 'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name, camera = self.data[idx].split()

        image = np.asarray(Image.open(os.path.join(self.data_dir, img_name)))  #imageio.imread(os.path.join(self.img_dir, img_name))
        gt_dict = target[camera]
        pred_target = float(gt_dict[self.quality_factor])

        if self.transform:
            image = self.transform(image)

        return image, pred_target

        
        
class FullImageDataset(Dataset):
    def __init__(self, label_file, data_dir, img_size, crop_num, quality_factor, transform=None):

        self.label_file = label_file
        self.data_dir = data_dir
        
        self.transform = transform
        self.img_size = img_size
        self.crop_num = crop_num
        self.quality_factor = quality_factor
        
        with open(self.label_file, 'r') as f:
            self.imgs = f.readlines()
            

    def __len__(self):
        return (len(self.imgs) * self.crop_num)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_idx = idx // self.crop_num
        img_name, camera = self.imgs[img_idx].split()

        image = np.asarray(Image.open(os.path.join(self.data_dir, img_name)))
        h, w, _ = image.shape
        
        hstart = random.randint(0, h-self.img_size)
        wstart = random.randint(0, w-self.img_size)
        img_crop = image[hstart:hstart+self.img_size, wstart:wstart+self.img_size, :]
                
        gt_dict = target[camera]
        pred_target = float(gt_dict[self.quality_factor])

        if self.transform:
            img_crop = self.transform(img_crop)

        return img_crop, pred_target
