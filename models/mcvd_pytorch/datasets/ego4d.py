import h5py
import numpy as np
import torch
import decord
import os

from torch.utils.data import Dataset, DataLoader
from torch import nn
from PIL import Image
from torchvision import transforms
from decord import VideoReader

FITVID_VAL_TRANSFORMS = transforms.Compose([transforms.Resize(64),
                      transforms.ToTensor(),])


class EGO4Dataset(Dataset):

    def __init__(self, clips, frames_per_sample=12, image_size=64, train=True, random_time=True, random_horizontal_flip=True,
                 total_videos=-1, skip_videos=0, with_target=True, is_color=True, modality='rgb', num_segments=1,
                 new_length=25, new_step=2, video_loader=False, use_decord=False):

        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.new_length = new_length

        self._new_step = new_step 
        self.video_loader = video_loader
        self.use_decord = use_decord
        self.transform = FITVID_VAL_TRANSFORMS
        
        self.clips = clips
        self.train = train
        self.frames_per_sample = frames_per_sample
        self.image_size = image_size
        self.random_time = random_time
        self.random_horizontal_flip = random_horizontal_flip
        self.total_videos = total_videos            # If we wish to restrict total number of videos (e.g. for val)
        self.with_target = with_target

        print(f"Dataset length: {self.__len__()}")

    def __len__(self):
        return len(self.data) if self.train else self.num_test_vids / 10
        
    def __getitem__(self, index):
        video_name = self.clips[index]

        if self.video_loader:
            try:
                decord_vr = decord.VideoReader(video_name, num_threads=1)
            except:
                return(self.__getitem__(index+1))

            duration = len(decord_vr)

        segment_indices, skip_offsets, new_step, skip_length = self._sample_train_indices(duration)

        images = self._video_TSN_decord_batch_loader(video_name, decord_vr, duration, 
                                                     segment_indices, skip_offsets, 
                                                     new_step, skip_length)

        process_data = torch.stack([self.transform(im) for im in images]) # T*C,H,W
        
        return process_data

    def _sample_train_indices(self, num_frames):
        new_step = self._new_step

        skip_length = self.new_length * new_step            
            
        average_duration = (num_frames - skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        skip_offsets = np.zeros(skip_length // new_step, dtype=int)
        return offsets + 1, skip_offsets, new_step, skip_length


    def _video_TSN_decord_batch_loader(self, video_name, video_reader, duration, indices, skip_offsets, new_step, skip_length):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, skip_length, new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + new_step < duration:
                    offset += new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, 
                                                                                                         video_name, 
                                                                                                         duration))
        return sampled_list

