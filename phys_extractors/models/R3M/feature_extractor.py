import argparse
import numpy as np
import os.path as osp
import os
import tqdm 
import h5py
import torch 

from r3m_model import FrozenPretrainedEncoder, load_model, pfR3M_LSTM_physion
from r3m_loader import R3MDataset
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict



def main(args):
    print('load model')
    model = FrozenPretrainedEncoder(args.encoder_name, 
                                    args.dynamics_name, 
                                    n_past=7, 
                                    full_rollout=True)
    model = load_model(model, args.model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    print('load data')
    dataset = R3MDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    data_size = len(dataset)
        
    n_features = 10
    # set up new dataset
    f = h5py.File(args.save_file, "w")
    dset1 = f.create_dataset("label", (data_size,), dtype='f')
    dset2 = f.create_dataset("observed", (data_size, 7, 2048), dtype='f')
    dset3 = f.create_dataset("observed_full_outcome", (data_size, 18, 2048), dtype='f')
    dset4 = f.create_dataset("simulation", (data_size, 18, 2048), dtype='f')

    print('start extraction')
    for i, batch in enumerate(tqdm.tqdm(loader)):
        v_in, label_in = batch
        v_in = v_in.to('cuda')
        
        if (i+1)*args.batch_size < data_size:
            dset1[i*args.batch_size:(i+1)*args.batch_size] = label_in.reshape(-1)
        else:
            dset1[i*args.batch_size:] = label_in.reshape(-1)
        # input is (Bs, T, 3, H, W)
        output = model(v_in)
        if (i+1)*args.batch_size < data_size:#(8, 4, 50, 16, 16) (8, 4, 50, 8, 8, 256) (8, 4, 5, 8, 8, 256)
            dset2[i*args.batch_size:(i+1)*args.batch_size] = output["input_states"].detach().cpu().numpy()
            dset3[i*args.batch_size:(i+1)*args.batch_size] = output["observed_states"].detach().cpu().numpy()
            dset4[i*args.batch_size:(i+1)*args.batch_size] = output["simulated_states"].detach().cpu().numpy()
        else:
            dset2[i*args.batch_size:] = output["input_states"].detach().cpu().numpy()
            dset3[i*args.batch_size:] = output["observed_states"].detach().cpu().numpy()
            dset4[i*args.batch_size:] = output["simulated_states"].detach().cpu().numpy()
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--encoder_name', type=str, required=True)
    parser.add_argument('--dynamics_name', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    main(args)

