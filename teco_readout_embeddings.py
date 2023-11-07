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
    
    # set up new dataset
    #f = h5py.File(args.save_file, "w")
    
    n_features = args.seq_len
    
#    dset1 = f.create_dataset("label", (data_size,), dtype='f')
    #if args.embeddings != 'h':
    #    dset2 = f.create_dataset("features_x", (data_size, n_features, 16, 16), dtype='f')
    #    dset3 = f.create_dataset("features_z", (data_size, n_features, 8, 8, 8), dtype='f')
    #    dset4 = f.create_dataset("features_h", (data_size, n_features - args.open_loop_ctx, 8, 8, 8), dtype='f')
    #else:
    ocp = []
    ocd = []
    ocd_focused = []
    labels = []
    contacts = []
    stimulus = []
#    dset2 = f.create_dataset("features", (data_size, 1, 8, 8, 256), dtype='f')

    for i, batch in enumerate(tqdm.tqdm(loader)):
        c_in = batch['collision_ind'].reshape(-1)
        contact_in = batch['contacts'].reshape(-1, 2)
        v_in = batch['video']
        act_in = batch['actions']
        label_in = batch['label'].reshape(-1)
        stim_in = batch['stimulus'].reshape(-1)
        
        labels += [label_in]
        contacts += [contact_in]
        stimulus.extend(stim_in)

        #if args.embeddings == 'h':
        h = readout_h_run(model, state, v_in, act_in, seed=args.seed)
        h = jax.device_get(h.squeeze(1))
        ocp += [h[:, :12].reshape(8, -1)]

        for i_c, c in enumerate(c_in):

            if c + 7 > 201:
                ocd_focused += [h[i_c,-12:].reshape(1, -1)]
            elif c - 6 < 0:
                ocd_focused += [h[i_c,:12].reshape(1, -1)]
            else:
                ocd_focused += [h[i_c,c-6:c+6].reshape(1, -1)]
            
            
            if c - 25 < 0:
                feats = h[i_c, :50]
            elif c + 25 > 201:
                feats = h[i_c, -50:]
            else:
                feats = h[i_c, c-25:c+25]
                
            if feats.shape[0] < 50:
                pad_width = ((0, 50 - feats.shape[1]), (0, 0), (0, 0), (0, 0))
                # Pad the array
                feats = np.pad(feats, pad_width, mode='constant', constant_values=0)
            ocd += [np.expand_dims(feats, axis=0).reshape(1, -1)]

    dt = h5py.special_dtype(vlen=str)

    print('save 1')
    with h5py.File(args.save_path_ocp ,'w') as hf:
        hf.create_dataset("features", data=np.concatenate(ocp))
        hf.create_dataset("label", data=np.concatenate(labels))
        hf.create_dataset("contacts", data=np.concatenate(contacts))  
        hf.create_dataset("filenames", data=stimulus, dtype=dt)
     
    print('save 2')
    with h5py.File(args.save_path_ocd ,'w') as hf:
        hf.create_dataset("features", data=np.concatenate(ocd))
        hf.create_dataset("label", data=np.concatenate(labels))
        hf.create_dataset("contacts", data=np.concatenate(contacts))  
        hf.create_dataset("filenames", data=stimulus, dtype=dt)
        
    print('save 3')
    with h5py.File(args.save_path_focused ,'w') as hf:
        hf.create_dataset("features", data=np.concatenate(ocd_focused))
        hf.create_dataset("label", data=np.concatenate(labels))
        hf.create_dataset("contacts", data=np.concatenate(contacts))  
        hf.create_dataset("filenames", data=stimulus, dtype=dt)
        '''   
        else:
            x, z, h = readout_z_run(model, state, v_in, act_in, seed=args.seed,
                                    scenario=args.scenario, seq_len=args.seq_len,
                                    open_loop_ctx=args.open_loop_ctx)
            # reshape
            x = x.reshape(-1, n_features, 16, 16)
            z = z.reshape(-1, n_features, 8, 8, 256)
            h = h.reshape(-1, n_features - args.open_loop_ctx, 8, 8, 256)
            # aggregate
            x = torch.tensor(x)
            x = nn.AdaptiveAvgPool2d((None, None))(x.float()) # --> 6400
            z = torch.tensor(z)
            z = nn.AdaptiveAvgPool3d(( 8, 8, 8))(z.float()) # --> 12800
            h = torch.tensor(h)
            h = nn.AdaptiveAvgPool3d(( 8, 8, 8))(h.float()) # --> 8704
            # add to data
            if (i+1)*args.batch_size < data_size:#(8, 4, 50, 16, 16) (8, 4, 50, 8, 8, 256) (8, 4, 5, 8, 8, 256)
                dset2[i*args.batch_size:(i+1)*args.batch_size] = x.numpy()
                dset3[i*args.batch_size:(i+1)*args.batch_size] = z.numpy()
                dset4[i*args.batch_size:(i+1)*args.batch_size] = h.numpy()
            else:
                dset2[i*args.batch_size:] = x.numpy()
                dset3[i*args.batch_size:] = z.numpy()
                dset4[i*args.batch_size:] = h.numpy()
    print('max_con ', con)
    f.close()
        '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--vqvae_ckpt', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path_ocp', type=str, required=True)
    parser.add_argument('--save_path_focused', type=str, required=True)
    parser.add_argument('--save_path_ocd', type=str, required=True)
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--scenario', type=str, default='past', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--open_loop_ctx', type=int, default=7)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--size', type=int)
    args = parser.parse_args()

    main(args)
