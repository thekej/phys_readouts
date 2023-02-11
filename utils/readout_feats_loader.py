import h5py
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch

class FeaturesDataset(Dataset):
    def __init__(self, h5_path, set_ = None):
        self.indices = set_
        data = h5py.File(h5_path)
        self.features = data['features']
        self.labels = data['label']


    def __len__(self):
        if not self.indices is None:
            return len(self.indices)
        return len(self.labels)

    def __getitem__(self, index):
        if not self.indices is None:
            index = self.indices[index]
        x = self.features[index]
        x = nn.AdaptiveAvgPool2d((1, 1))(torch.tensor(x))
        y = self.labels[index].astype(np.float32)
        return x, y