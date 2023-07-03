import argparse
import numpy as np
import os.path as osp
import os
import tqdm 
import h5py
import torch 

from fitvid import FitVid
from fitvid_loader import ReadoutDataset
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

def load_model(
    model, model_path, state_dict_key="state_dict"
):
    params = torch.load(model_path, map_location="cpu")
    
    sd = params
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith("module."):
            name = k[7:]  # remove 'module.' of dataparallel/DDP
        else:
            name = k
        new_sd[name] = v
    model.load_state_dict(new_sd)
    print(f"Loaded parameters from {model_path}")

    # Set model to eval mode'''
    model.eval()

    return model

def main(args):
    print('load model')
    model = FitVid(n_past=args.n_past, train=False)
    model = load_model(model, args.model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    print('load data')
    dataset = ReadoutDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    data_size = len(loader)#5608
    #for b in loader:
    #    v_in, label_in = b
    #    data_size += v_in.shape[0]
        
    n_features = 10
    # set up new dataset
    f = h5py.File(args.save_file, "w")
    dset1 = f.create_dataset("label", (data_size,), dtype='f')
    dset2 = f.create_dataset("features", (data_size, 24, 128), dtype='f')

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
        out = output['h_preds']
        if (i+1)*args.batch_size < data_size:#(8, 4, 50, 16, 16) (8, 4, 50, 8, 8, 256) (8, 4, 5, 8, 8, 256)
            dset2[i*args.batch_size:(i+1)*args.batch_size] = out.detach().cpu().numpy()
        else:
            dset2[i*args.batch_size:] = out.detach().cpu().numpy()
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--n-past', type=int, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    main(args)

