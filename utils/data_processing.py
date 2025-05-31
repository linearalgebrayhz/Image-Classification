# data located in /scratch/network/hy4522/DL_data/fruits-360_100x100/fruits-360
import argparse
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import numpy as np

class FruitImageDataset(Dataset):
    
    def __init__(self,root_dir, transform = None):
        
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        classes = sorted(os.listdir(root_dir))
        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)
            fruit_name = " ".join(class_name.split()[:-1])
            if not os.path.isdir(class_dir):
                continue
            new_idx = idx
            if fruit_name in self.class_to_idx.keys():
                new_idx = self.class_to_idx[fruit_name]
            else:
                self.class_to_idx[class_name] = idx
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.lower().endswith(('.png','.jpg','.jpeg')):
                    self.image_paths.append(img_path)
                    self.labels.append(new_idx)
                    
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    # for testing
    def get_class_mapping(self):
        return self.class_to_idx.copy()
    
if __name__ == "__main__":
    train_dataset = FruitImageDataset(
        root_dir="/scratch/network/hy4522/DL_data/fruits-360_100x100/fruits-360/Training",
        transform = transforms.ToTensor()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle = True,
        num_workers= 0
    )
    
    print(f"训练集类别数: {len(train_dataset.class_to_idx)}")
    
    # 让GPT帮忙写的hh
    import matplotlib.pyplot as plt
    
    def imshow(inp, title = None):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        plt.imshow(inp)
        plt.imsave("./sample.png", inp)
        if title: plt.title(title)
    
    # 获取一个batch数据
    inputs, classes = next(iter(train_loader))
    out = torchvision.utils.make_grid(inputs)

    # 显示带标签的图像
    imshow(out, title=[train_dataset.idx_to_class[x.item()] for x in classes])
    
# DataLoader的工作原理，几个参数的作用，比如num_workers， pin_memory, collate_fn