import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf

from intphys_evaluator.feature_extract_interface import IntPhysFeatureExtractor

from collections import OrderedDict
from torchvision import transforms

import torch.nn.functional as F


from models.mcvd_pytorch.load_model_from_ckpt import load_model, get_readout_sampler, init_samples
from models.mcvd_pytorch.datasets import data_transform
from models.mcvd_pytorch.runners.ncsn_runner import conditioning_fn
    
class MCVD(IntPhysFeatureExtractor):
    def __init__(self, weights_path, 
                 model_type='physion',
                 n_features=13):
        super().__init__()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # load model
        self.scorenet, self.config = load_model(weights_path, device)
        # get sampler
        self.sampler = get_readout_sampler(self.config)
        self.n_features = n_features
        self.model_type = model_type
        
    def transform(self):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        return transform

    def get_plausibility(self, videos , inp):
        #videos = torch.stack([self.transform_video_tensor(vid) for vid in videos])
        input_frames = data_transform(self.config, videos)
        if self.config.data.num_frames_cond+self.config.data.num_frames > videos.shape[1]:
            added_frames = self.config.data.num_frames_cond+self.config.data.num_frames - videos.shape[1]
            input_frames = torch.cat([input_frames] + [input_frames[:, -1].unsqueeze(1)]*added_frames, axis=1)
            
        output = []
        for j in range(0, videos.shape[1] - (self.config.data.num_frames_cond+self.config.data.num_frames)):
            if j + self.config.data.num_frames_cond+self.config.data.num_frames > videos.shape[1]:
                break
            real, cond, cond_mask = conditioning_fn(self.config, 
                                                    input_frames[:, 
                                                    j:j+self.config.data.num_frames_cond+self.config.data.num_frames, 
                                                    :, :, :], 
                                        num_frames_pred=self.config.data.num_frames,
                                        prob_mask_cond=getattr(self.config.data, 'prob_mask_cond', 0.0),
                                        prob_mask_future=getattr(self.config.data, 'prob_mask_future', 0.0))

            init = init_samples(len(real), self.config)
            with torch.no_grad():
                pred, gamma, beta, mid = self.sampler(init, self.scorenet, cond=cond,
                                         cond_mask=cond_mask,
                                         subsample=100, verbose=True)
            output +=  [pred.reshape(-1, 3, 64, 64)]
        
        # Define the Mean Squared Error loss function (reduction='none' keeps the individual losses)
        mse_loss = nn.MSELoss(reduction='none')

        # Compute the reconstruction loss per image
        ground_truth = videos[:, self.config.data.num_frames_cond+self.config.data.num_frames:]
        N, T, _, _, _ = ground_truth.shape
        loss_per_image = mse_loss(torch.stack(output, axis=1), ground_truth)

        # Sum the loss over the spatial dimensions and channels
        loss_per_image = loss_per_image.view(N, T, -1).mean(dim=-1)

        # Find the maximum reconstruction loss among all images
        plausibility = loss_per_image.max(dim=1)[0]
        plausibility_numpy = plausibility.cpu().numpy()
        return 1 / plausibility_numpy

class MCVD_PHYSION(MCVD):
    def __init__(self, weights_path):
        super().__init__(weights_path, model_type='physion')
        
class MCVD_EGO4D(MCVD):
    def __init__(self, weights_path):
        super().__init__(weights_path, model_type='ego4d')
        
class MCVD_UCF(MCVD):
    def __init__(self, weights_path):
        super().__init__(weights_path, model_type='ucf')


class FITVID(IntPhysFeatureExtractor):
    # input is (Bs, T, 3, H, W)
    def __init__(self, weights_path, n_past=7):
        super().__init__()
        from models.FitVid import fitvid
        
        # IMPORTANT: n_past decides if OCP or OCD
        model = fitvid.FitVid(n_past=n_past, train=False)
        params = torch.load(weights_path, map_location="cpu")

        new_sd = OrderedDict()
        for k, v in params.items():
            name = k[7:] if k.startswith("module.") else k
            new_sd[name] = v
        model.load_state_dict(new_sd)
        print(f"Loaded parameters from {weights_path}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        
    def transform(self):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        return transform

    def get_plausibility(self, videos, inp): 
        videos = videos.to(self.device)
        with torch.no_grad():
            output = self.model(videos, n_past=videos.shape[1])
        preds = output['preds'][:, 1:] # first frame is a black frame

        # Define the Mean Squared Error loss function (reduction='none' keeps the individual losses)
        mse_loss = nn.MSELoss(reduction='none')

        # Compute the reconstruction loss per image
        ground_truth = videos[:, 1:]
        N, T, _, _, _ = ground_truth.shape
        loss_per_image = mse_loss(preds, ground_truth)

        # Sum the loss over the spatial dimensions and channels
        loss_per_image = loss_per_image.view(N, T, -1).mean(dim=-1)

        # Find the maximum reconstruction loss among all images
        plausibility = loss_per_image.max(dim=1)[0]
        plausibility_numpy = plausibility.cpu().numpy()
        return 1 / plausibility_numpy

import argparse
import numpy as np
import os.path as osp
import os
import jax
import yaml
import pickle
import tqdm 
import h5py
import torch

from torch import nn

from models.teco.teco.train_utils import seed_all
from models.teco.teco.models import load_ckpt, sample_plausibility
from models.teco.teco.data import Data

class TECO(IntPhysFeatureExtractor):
    # input is (Bs, T, 3, H, W)
    def __init__(self, weights_path, n_past=7):
        super().__init__()


        

        seed_all(0)
        
        kwargs = dict()
        kwargs['batch_size'] = 7
        kwargs['open_loop_ctx'] = 100#args.open_loop_ctx
        kwargs['seq_len'] = 100
        print('load model')
        model, state, config = load_ckpt(weights_path, return_config=True, 
                                         **kwargs, data_path='/ccn2/u/thekej/intphys_teco/intphys_encoded.h5py',
                                         vqvae_ckpt='/ccn2/u/thekej/kinetics_vqgan/')
        self.encoded = h5py.File('/ccn2/u/thekej/intphys_teco/intphys_encoded.h5py') # Get encoded 
        self.encoded_filename = [s.decode() for s in self.encoded['filenames'][:]]

        self.model = model
        self.state = state
        
    def transform(self):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        return transform

    def get_plausibility(self, videos, filenames): 
        actions = np.zeros((videos.shape[0],), dtype=np.int32)
    
        # If you want to convert these numpy arrays to tf.Tensor
        encoded_videos = []
        for f in filenames:
            ind = self.encoded_filename.index(f)
            encoded_videos += [self.encoded['video'][ind]]

        encoded_videos = np.stack(encoded_videos)
        print('inpu: ', encoded_videos.shape)
        encoded_videos = encoded_videos.reshape(1, 1, encoded_videos.shape[1], 
                                                encoded_videos.shape[2], encoded_videos.shape[3])
        
        preds = sample_plausibility(self.model, self.state, encoded_videos, actions, seed=0)

        # Define the Mean Squared Error loss function (reduction='none' keeps the individual losses)
        mse_loss = nn.MSELoss(reduction='none')

        # Compute the reconstruction loss per image
        ground_truth = videos[:, 1:]
        N, T, _, _, _ = ground_truth.shape
        loss_per_image = mse_loss(preds[:, :-1], ground_truth)

        # Sum the loss over the spatial dimensions and channels
        loss_per_image = loss_per_image.view(N, T, -1).mean(dim=-1)

        # Find the maximum reconstruction loss among all images
        plausibility = loss_per_image.max(dim=1)[0]
        plausibility_numpy = plausibility.cpu().numpy()
        return 1 / plausibility_numpy