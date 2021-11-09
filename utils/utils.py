import os
import pandas as pd
import numpy as np
import pickle
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

def get_dataframe(path):
    img_dir_normal = os.listdir(os.path.join(path, 'NORMAL'))
    label_normal = np.zeros((len(img_dir_normal)))

    img_dir_sick = os.listdir(os.path.join(path, 'PNEUMONIA'))
    label_sick = np.ones((len(img_dir_sick))) 
    
    add_normal = lambda x: os.path.join(os.path.join(path, 'NORMAL'), x)
    add_sick = lambda x: os.path.join(os.path.join(path, 'PNEUMONIA'), x)
    
    img_dir_normal = list(map(add_normal, img_dir_normal))
    img_dir_sick = list(map(add_sick, img_dir_sick))
    
    img_dir = img_dir_normal + img_dir_sick
    label = np.concatenate((label_normal, label_sick))
    
    data = {'Image Dir': img_dir, 'Label': label}
    df = pd.DataFrame(data)
    
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    return df

def get_dataframe_undersampling(path):
    img_dir_normal = os.listdir(os.path.join(path, 'NORMAL'))
    label_normal = np.zeros((len(img_dir_normal)))

    img_dir_sick = os.listdir(os.path.join(path, 'PNEUMONIA'))
    label_sick = np.ones((len(img_dir_sick))) 
    
    img_dir_sick = img_dir_sick[:1300]
    label_sick = label_sick[:1300]
    
    add_normal = lambda x: os.path.join(os.path.join(path, 'NORMAL'), x)
    add_sick = lambda x: os.path.join(os.path.join(path, 'PNEUMONIA'), x)
    
    img_dir_normal = list(map(add_normal, img_dir_normal))
    img_dir_sick = list(map(add_sick, img_dir_sick))
    
    img_dir = img_dir_normal + img_dir_sick
    label = np.concatenate((label_normal, label_sick))
    
    data = {'Image Dir': img_dir, 'Label': label}
    df = pd.DataFrame(data)
    
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    return df

def transform_img(img, transform):
    img = transform(img).unsqueeze(0)
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'
    return img