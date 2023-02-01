import h5py
import numpy as np

from torch.utils.data import Dataset, DataLoader


class FeaturesDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as hf:
            self.features = hf['mid_embed'][:]
            self.labels = np.zeros((hf['mid_embed'].shape[0], 1))#hf['labels'][:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index].astype(np.float32)
        return x, y