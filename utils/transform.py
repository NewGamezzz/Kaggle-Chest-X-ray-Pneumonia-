import os
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from utils.utils import *

class MorphTransform:
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def __call__(self, img):
        img = np.array(img)
        kernel = np.ones((self.kernel_size, self.kernel_size),np.uint8)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        top = cv2.subtract(img, opening)

        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        bot = cv2.subtract(closing, img)

        img_enh = cv2.subtract(cv2.add(img, top), bot)
        img_enh = np.expand_dims(img_enh, axis=0)
        return torch.FloatTensor(img_enh/255.0)
    
class CADTransform:
    def __init__(self, r=9, var=75, axis=0):
        self.r = r
        self.var = var
        self.axis = axis

    def __call__(self, img):
        img = np.array(img)
        equalized = cv2.equalizeHist(img)
        bilateral = cv2.bilateralFilter(img, self.r, self.var, self.var)
        R = np.expand_dims(equalized, axis=self.axis)
        G = np.expand_dims(img, axis=self.axis)
        B = np.expand_dims(bilateral, axis=self.axis)

        img_rgb = np.concatenate([R, G, B], axis=self.axis)
        
        return torch.FloatTensor(img_rgb/255.0)
    
