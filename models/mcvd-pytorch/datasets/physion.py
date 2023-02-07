# https://github.com/edenton/svg/blob/master/data/kth.py
import numpy as np
import os
import pickle
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .h5 import HDF5Dataset


class PhysionDataset(Dataset):

    def __init__(self, data_path, frames_per_sample=5, image_size=64, train=True, random_time=True, random_horizontal_flip=True,
                 total_videos=-1, skip_videos=0, with_target=True, complete=False, simulation=False):

        self.data_path = data_path                    # '/path/to/Datasets/UCF101_64_h5' (with .hdf5 file in it), or to the hdf5 file itself
        self.train = train
        self.frames_per_sample = frames_per_sample
        self.image_size = image_size
        self.random_time = random_time
        self.random_horizontal_flip = random_horizontal_flip
        self.total_videos = total_videos            # If we wish to restrict total number of videos (e.g. for val)
        self.with_target = with_target
        self.complete = complete
        self.simulation = simulation
        
        # Read h5 files as dataset
        self.videos_ds = HDF5Dataset(self.data_path)

        # Train
        # self.num_train_vids = 9624
        # self.num_test_vids = 3696   # -> 369 : https://arxiv.org/pdf/1511.05440.pdf takes every 10th test video
        with self.videos_ds.opener(self.videos_ds.shard_paths[0]) as f:
            self.num_train_vids = f['num_train'][()]
            self.num_test_vids = f['num_test'][()]//10  # https://arxiv.org/pdf/1511.05440.pdf takes every 10th test video
        
        self.indices = []
        for v in range(self.num_train_vids):
            shard_idx, idx_in_shard = self.videos_ds.get_indices(v)
            if self.complete or self.simulation:
                self.indices += [(shard_idx, idx_in_shard)]
            with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
                if f['len'][str(idx_in_shard)][()] >= frames_per_sample and not self.complete and not self.simulation:
                    self.indices += [(shard_idx, idx_in_shard)]

        print(f"Dataset length: {self.__len__()}")

    def len_of_vid(self, index):
        video_index = index % self.__len__()
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx_in_shard)][()]
        return video_len

    def __len__(self):
        return len(self.indices)

    def max_index(self):
        return self.num_train_vids if self.train else self.num_test_vids

    def __getitem__(self, index, time_idx=0):

        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video
        data_transform = transforms.Compose([
            transforms.ToTensor()
            ])
        shard_idx, idx_in_shard = self.indices[index]

        # read data
        prefinals = []
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            target = int(f['target'][str(idx_in_shard)][()])
            # slice data
            video_len = f['len'][str(idx_in_shard)][()]
            if self.complete or self.simulation:
                if video_len < self.frames_per_sample:
                    for i in range(video_len):
                        img = f[str(idx_in_shard)][str(i)][()]
                        prefinals.append(data_transform(img))
                    prefinals += [data_transform(img)] * (self.frames_per_sample - video_len)
                else:
                    for i in range(self.frames_per_sample):
                        img = f[str(idx_in_shard)][str(i)][()]
                        prefinals.append(data_transform(img))
            else:
                for i in range(self.frames_per_sample):
                    img = f[str(idx_in_shard)][str(i)][()]
                    prefinals.append(data_transform(img))

        video = torch.stack(prefinals)

        if self.with_target:
            return video, torch.tensor(target)
        else:
            return video
