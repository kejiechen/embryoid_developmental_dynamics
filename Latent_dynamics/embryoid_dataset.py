import os
import datetime
from shutil import copyfile
from collections import namedtuple, defaultdict
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics._regression import mean_squared_error as mse
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import consts
import pdb


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y, n = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image_resize = np.zeros((n, self.output_size[0], self.output_size[1]), dtype="uint8")
            for ni in range(n):
                image_resize[ni,:,:] = zoom(image[:,:,ni], (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            if len(np.shape(label))>0:
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image_resize.astype(np.float32))
        if len(np.shape(label))>0:
            label = torch.from_numpy(label.astype(np.float32))
            sample = {'image': image, 'label': label.long()}
        else:
            sample = {'image': image}
        return sample


class Embryoid_dataset(Dataset):
    def __init__(self, base_dir, sample_name_list, transform=None):
        self.transform = transform  # using transform in torch!
        self.sample_list = sample_name_list
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx]
        data_path = os.path.join(self.data_dir, slice_name)
        data = np.load(data_path, allow_pickle=True)
        image, label = data['image'], data['mask']

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx]
        return sample
