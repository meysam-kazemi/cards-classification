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
BATCH_SIZE = 64

TRANSFORM_IMG = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
])

