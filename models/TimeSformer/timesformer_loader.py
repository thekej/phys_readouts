import h5py
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch

class TimesformerDataset(Dataset):
    def __init__(self, h5_path):
        data = h5py.File(h5_path)
        self.videos = data['video']
        self.labels = data['label']


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.videos[index]
        observed_list = [x[0], x[0]] + [a[0] for a in np.array_split(x[:6], 6)]
        observed = np.stack(observed_list)
        xs = [observed.transpose(1, 0, 2, 3)] + [a.transpose(1, 0, 2, 3) for a in np.array_split(x[6:30], 3)]
        y = self.labels[index]
        return np.stack(xs), y