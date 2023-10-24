import argparse
import sys, os
import glob, os
import mediapy as media
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import nn

from load_model_from_ckpt import load_model, get_readout_sampler, init_samples
from datasets import get_dataset, data_transform, inverse_data_transform
from datasets.physion import PhysionDataset
from runners.ncsn_runner import conditioning_fn

from os.path import expanduser
home = expanduser("~")

'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python extract_mcvd.py --readout_type PAST --video_length 4 --stimuli_length 12 --save_file test.hdf5 --data_path /ccn2/u/thekej/phys_readouts_mcvd_debug/

python datasets/physion_convert.py --out_dir /ccn2/u/thekej/phys_readouts_mcvd_debug/ --ucf_dir /ccn2/u/rmvenkat/data/testing_physion/test_data_felix_with_map/test_data_balanced/ --map_dir /ccn2/u/thekej/phys_readouts_mcvd_debug/stimulus_map.json

'''

GENERATION_LENGTH = 20

PRED_LENGTH = 4

def get_dataset(args, config):
    
    frames_per_sample = 25#args.video_length + args.stimuli_length
    dataset = PhysionDataset(args.data_path, frames_per_sample=frames_per_sample, 
                             image_size=config.data.image_size, train=False, random_time=False,
                             random_horizontal_flip=False,
                             complete=args.readout_type == 'COMPLETE',
                             simulation=args.readout_type == 'SIMULATION') #change this

    return dataset

def extract(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ckpt_path = glob.glob(os.path.join(args.model_path, "checkpoint_*.pt"))[0]
    
    # load model
    scorenet, config = load_model(ckpt_path, device)
    
    # get sampler
    sampler = get_readout_sampler(config)

    # load data
    dataset = get_dataset(args, config)

    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=config.data.num_workers, drop_last=False)
    data_size = 0
    for t, label in test_loader:
        data_size += t.shape[0]
    print('Dataset size: ', data_size)
    # 
    if args.readout_type == 'SIMULATION':
        n_features = 1 + args.video_length // PRED_LENGTH
    elif args.readout_type == 'PAST':
        n_features = 1
    elif args.readout_type == 'COMPLETE':
        n_features = 1 + args.video_length // PRED_LENGTH
        
    print('n features: ',n_features)
        
    # if gamma beta 288, 64, 64 if mid 1152, 8, 8
    #if not args.latent_type == 'middle_embeds':
    #    f1, f2, f3 = 288, 2, 2
    #else:
    #    f4 = 1152 
    f1, f2, f3, f4 = 288, 2, 2, 1152

    # set up new dataset
    f = h5py.File(args.save_file, "w")
    dset2 = f.create_dataset("label", (data_size,), dtype='f')
    dset4 = f.create_dataset("features_mid", (data_size, n_features, f4, 1, 1), dtype='f')

    # extract features
    for i, (test_x, label) in enumerate(test_loader):
        input_frames = data_transform(config, test_x)
        #if not args.latent_type == 'middle_embeds':
        features_array = {'mid': np.zeros((test_x.shape[0], n_features, f4, 1, 1)),
                             }
        #else:
        #    features_array = np.zeros((test_x.shape[0], n_features, f1, f2, f3))
        
        real, cond, cond_mask = conditioning_fn(config, input_frames[:, 8:16, :, :, :], num_frames_pred=config.data.num_frames,
                                            prob_mask_cond=getattr(config.data, 'prob_mask_cond', 0.0),
                                            prob_mask_future=getattr(config.data, 'prob_mask_future', 0.0))
        init = init_samples(len(real), config)
        import time
        t = time.time()
        pred, gamma, beta, mid = sampler(init, scorenet, cond=cond, cond_mask=cond_mask, subsample=args.sub, verbose=True)
        print('one processing: ', time.time() - t)
        #if args.latent_type == 'middle_embeds':
        mid = nn.AdaptiveAvgPool3d((None, 1, 1))(mid.cpu())
        features_array['mid'][:, 0, :, :, :] =  mid.numpy()
            
        for j in range(1, n_features):
            if args.readout_type == 'SIMULATION':
                if 4*j < args.stimuli_length:
                    real, cond, cond_mask = conditioning_fn(config, input_frames[:, 8+4*j:4*j+16, :, :, :], 
                                                        num_frames_pred=config.data.num_frames,
                                            prob_mask_cond=getattr(config.data, 'prob_mask_cond', 0.0),
                                            prob_mask_future=getattr(config.data, 'prob_mask_future', 0.0))
                else:
                    cond = pred
            elif args.readout_type == 'COMPLETE':
                real, cond, cond_mask = conditioning_fn(config, input_frames[:, 4*j:4*j+8, :, :, :], 
                                                        num_frames_pred=config.data.num_frames,
                                            prob_mask_cond=getattr(config.data, 'prob_mask_cond', 0.0),
                                            prob_mask_future=getattr(config.data, 'prob_mask_future', 0.0))
                    
            else:
                cond = pred
            init = init_samples(len(real), config)
            pred, gamma, beta, mid = sampler(init, scorenet, cond=cond, cond_mask=cond_mask, subsample=args.sub, verbose=True)
            #if args.latent_type == 'middle_embeds':
            mid = nn.AdaptiveAvgPool3d((None, 1, 1))(mid.cpu())
            features_array['mid'][:, j, :, :, :] =  mid.numpy()
            #else:
            
        print('whole processing: ', time.time() - t)
        if pred.shape[0] == args.batch_size:
            dset2[i * args.batch_size: (i+1)*args.batch_size] = label
            dset4[i * args.batch_size: (i+1)*args.batch_size, :, :, :, :] = features_array['mid']
        else:
            dset2[i * args.batch_size:] = label
            dset4[i * args.batch_size:, :, :, :, :] = features_array['mid']
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, 
                        required=True, help="Path to the data")
    parser.add_argument("--model_path", type=str, 
                        default = "/ccn2/u/thekej/ucf10132_big192_288_4c4_unetm_spade/logs/",
                        help="Path to the model")
    parser.add_argument("--readout_type", type=str, choices=["SIMULATION", "PAST", "COMPLETE"], 
                        help="Readout type")
    parser.add_argument("--latent_type", type=str, choices=["middle_embeds", "gamma", "beta"], 
                        default='middle_embeds', help="Latent type")
    parser.add_argument("--video_length", type=int, default=20, help="Length of the video in frames")
    parser.add_argument("--stimuli_length", type=int, default=4, help="Length of the stimuli in frames")
    parser.add_argument("--batch_size", type=int, default=256, help="Length of the stimuli in frames")
    parser.add_argument("--sub", type=int, default=100, help="Length of the stimuli in frames")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--save_file", type=str, default='file.hdf5', help="Path to the output hdf5")
    
    args = parser.parse_args()
    extract(args)
