import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision 
from torchvision import transforms
import matplotlib.pyplot as plt

# Configuration variables
TRAIN_DATA_PATH = "./cards/train/"
TEST_DATA_PATH = "./cards/test/"
VALID_DATA_PATH = "./cards/valid/"
DATAFRAME_PATH = "./cards/cards.csv"
BATCH_SIZE = 64

TRANSFORM_IMG = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
])

# Load Data
train = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_loader  = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid = torchvision.datasets.ImageFolder(VALID_DATA_PATH, transform=TRANSFORM_IMG)
valid_loader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Classes
df = pd.read_csv(DATAFRAME_PATH)
classes = []
for i in range(53):
  classes.append((df[df['class index']==i]['labels']).iloc[0])

# Save Classes
with open("classes.txt", "w") as f:
  for c in classes:
    f.write(c+"\n")
