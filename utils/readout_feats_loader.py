import h5py
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch

class FeaturesDataset(Dataset):
    def __init__(self, h5_path, set_ = None, scenario='complete'):
        self.indices = set_
        self.scenario = scenario
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
        if self.scenario == 'past':
            x = x[0]
        y = self.labels[index].astype(np.float32)
        return x, y
