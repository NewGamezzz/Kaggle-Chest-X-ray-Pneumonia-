import os
import pandas as pd
import numpy as np
import pickle
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from utils.utils import *

class XRayDataset(Dataset):
    def __init__(self, data, transform=None): ## add max_sentence
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data.iloc[idx, 0])
        image = Image.open(img_path).resize((256, 256))
        image = image.convert('L')
        
        if self.transform != None:
            image = self.transform(image)
        
        label = self.data.iloc[idx, 1]
        label = torch.tensor(label, dtype=torch.long)
        return image, label