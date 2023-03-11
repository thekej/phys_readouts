import h5py
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch

class R3MDataset(Dataset):
    def __init__(self, h5_path):
        data = h5py.File(h5_path)
        self.videos = data['video']
        self.labels = data['label']


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.videos[index]
        y = self.labels[index]
        return x, y
    
class R3MTrainDataset(Dataset):
    def __init__(self, h5_path, indices):
        data = h5py.File(h5_path)
        self.videos = data['video']
        self.indices = indices


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        x = self.videos[index]
        return x