import argparse
import numpy as np
import os.path as osp
import os
import jax
import yaml
import pickle
import tqdm 
import h5py

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
    data_size = 9993
    
    # set up new dataset
    f = h5py.File(args.save_file, "w")
    
    n_features = args.seq_len
    
    dset1 = f.create_dataset("label", (data_size,), dtype='f')
    if args.embeddings != 'h':
        dset2 = f.create_dataset("features_x", (data_size, n_features, 16, 16), dtype='f')
        dset3 = f.create_dataset("features_z", (data_size, n_features, 8, 8, 256), dtype='f')
        dset4 = f.create_dataset("features_h", (data_size, n_features - args.open_loop_ctx, 8, 8, 256), dtype='f')
    else:
        dset2 = f.create_dataset("features", (data_size, 1, 8, 8, 256), dtype='f')


    for i, batch in enumerate(tqdm.tqdm(loader)):
        v_in = batch['video']
        act_in = batch['actions']
        label_in = batch['label']

        if (i+1)*args.batch_size < data_size:
            dset1[i*args.batch_size:(i+1)*args.batch_size] = label_in.reshape(-1)
        else:
            dset1[i*args.batch_size:] = label_in.reshape(-1)

        if args.embeddings == 'h':
            h = readout_h_run(model, state, v_in, act_in, seed=args.seed)
            if (i+1)*args.batch_size < data_size:
                dset2[i*args.batch_size:(i+1)*args.batch_size]  = h.reshape(-1, 1, 8, 8, 256)
            else:
                dset2[i*args.batch_size:]  = h.reshape(-1, 1, 8, 8, 256)
            
        else:
            x, z, h = readout_z_run(model, state, v_in, act_in, seed=args.seed,
                                    scenario=args.scenario, seq_len=args.seq_len,
                                    open_loop_ctx=args.open_loop_ctx)
            # reshape
            x = x.reshape(-1, n_features, 16, 16)
            z = z.reshape(-1, n_features, 8, 8, 256)
            h = h.reshape(-1, n_features - args.open_loop_ctx, 8, 8, 256)
            # aggregate
            x = nn.AdaptiveAvgPool2d((1, 1))(torch.tensor(x))
            z = nn.AdaptiveAvgPool3d((None, None, 1))(torch.tensor(z))
            h = nn.AdaptiveAvgPool3d((None, None, 1))(torch.tensor(h))
            # add to data
            if (i+1)*args.batch_size < data_size:#(8, 4, 50, 16, 16) (8, 4, 50, 8, 8, 256) (8, 4, 5, 8, 8, 256)
                dset2[i*args.batch_size:(i+1)*args.batch_size] = x.numpy()
                dset3[i*args.batch_size:(i+1)*args.batch_size] = z.numpy()
                dset4[i*args.batch_size:(i+1)*args.batch_size] = h.numpy()
            else:
                dset2[i*args.batch_size:] = x.numpy()
                dset3[i*args.batch_size:] = z.numpy()
                dset4[i*args.batch_size:] = h.numpy()

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--vqvae_ckpt', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--scenario', type=str, default='past', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=45)
    parser.add_argument('--open_loop_ctx', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    main(args)
