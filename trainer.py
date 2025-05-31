import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from models import *
from utils.data_processing import FruitImageDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"current device is {device}")

parser = argparse.ArgumentParser(
    prog='trainer.py',
    description="training classification model",
    epilog=""
)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

train_dataset = FruitImageDataset(
    root_dir="/scratch/network/hy4522/DL_data/fruits-360_100x100/fruits-360/Training", 
    transform=train_transform
)

test_dataset = FruitImageDataset(
    root_dir="/scratch/network/hy4522/DL_data/fruits-360_100x100/fruits-360/Test",
    transform=test_transform
)

class_mapping = train_dataset.get_class_mapping()
num_classes = len(class_mapping)
batch_size = 32
train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle = False,
    num_workers=0   
)

model = ResNet(3, num_classes=num_classes).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

def train(dataloader, batch_size, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader), total = size//batch_size, desc="Training classification"):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        # pred = F.softmax(pred) no need!
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"\ntraining loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)
            
epochs = 3
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader,batch_size, model, loss_fn, optimizer)