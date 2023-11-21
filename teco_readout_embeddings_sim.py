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
from models.teco.teco.models import load_ckpt, readout_z_run, readout_h_run
from models.teco.teco.data import Data


def main(args):
    seed_all(args.seed)

    kwargs = dict()
    if args.batch_size is not None:
        kwargs['batch_size'] = args.batch_size
    if args.open_loop_ctx is not None:
        kwargs['open_loop_ctx'] = args.open_loop_ctx
    kwargs['seq_len'] = args.seq_len
    print('load model')
    model, state, config = load_ckpt(args.ckpt, return_config=True, 
                                     **kwargs, data_path=args.data_path,
                                     vqvae_ckpt=args.vqvae_ckpt)

    #config = pickle.load(open(osp.join(args.ckpt, 'args'), 'rb'))
    for k, v in kwargs.items():
        setattr(config, k, v)

    print('load data')
    
    config.data_path = args.data_path
    data = Data(config)
    loader = data.create_iterator(train=False, prefetch=False, repeat=False)
    data_size = args.size
    
    n_features = args.seq_len
    
    simulation = []
    labels = []
    contacts = []
    stimulus = []

    for i, batch in enumerate(tqdm.tqdm(loader)):
        c_in = batch['collision_ind'].reshape(-1)
        contact_in = batch['contacts'].reshape(-1, 2)
        v_in = batch['video']
        #print(v_in.shape)
        v_in = v_in.reshape(-1, v_in.shape[2], v_in.shape[3],  v_in.shape[4])
        act_in = batch['actions']
        label_in = batch['label'].reshape(-1)
        stim_in = batch['stimulus'].reshape(-1)
        
        labels += [label_in]
        contacts += [contact_in]
        stimulus.extend(stim_in)
        
        v_in_processed = []
        for i_c, c in enumerate(c_in):
            if c - 25 < 0:
                v_entry = v_in[i_c, :50]
            elif c + 25 > 201:
                v_entry = v_in[i_c, -50:]
            else:
                v_entry = v_in[i_c, c-25:c+25]
                
            if v_entry.shape[0] < 50:
                pad_width = ((0, 50 - v_in.shape[0]), (0, 0), (0, 0))
                # Pad the array
                v_entry = np.pad(v_entry, pad_width, mode='constant', constant_values=0)

            v_in_processed += [v_entry]

        #if args.embeddings == 'h':
        v_in_processed = np.stack(v_in_processed)
        v_in = v_in_processed.reshape(8, -1, 
                                      v_in_processed.shape[1], 
                                      v_in_processed.shape[2],  
                                      v_in_processed.shape[3])

        h = readout_h_run(model, state, v_in, act_in, seed=args.seed)
        h = jax.device_get(h)

        h_sim = readout_z_run(model, state, v_in, act_in, seed=args.seed, 
                          state_spec=None, scenario='simulation', seq_len=50, 
                          open_loop_ctx=12)

        h_sim = jax.device_get(h_sim)

        feats = np.concatenate([h[:, :, :12], h_sim], axis=2)
        simulation += [feats.squeeze(1)]

    dt = h5py.special_dtype(vlen=str)

    print('save 1')
    with h5py.File(args.save_path ,'w') as hf:
        hf.create_dataset("features", data=np.concatenate(simulation))
        hf.create_dataset("label", data=np.concatenate(labels))
        hf.create_dataset("contacts", data=np.concatenate(contacts))  
        hf.create_dataset("filenames", data=stimulus, dtype=dt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--vqvae_ckpt', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--scenario', type=str, default='past', required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--open_loop_ctx', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--size', type=int)
    args = parser.parse_args()

    main(args)
