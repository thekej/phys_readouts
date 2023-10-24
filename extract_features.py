import argparse
import numpy as np
import os.path as osp
import os
import tqdm 
import h5py
import torch 

from models import Extractor
from fitvid_loader import ReadoutDataset
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

def main(args):
    print('load model')
    model = Extractor(args.model_name,
                      args.model_path,
                      n_past=args.n_past,
                      model_type = args.model_type,
                      n_features = args.n_features
                     )

    print('load data')
    dataset = ReadoutDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    data_size = len(loader)#5608
    #for b in loader:
    #    v_in, label_in = b
    #    data_size += v_in.shape[0]
        
    n_features = 10
    # set up new dataset
    label_dset = []
    contacts_dset = []
    features_dset = []

    print('start extraction')
    for i, batch in enumerate(tqdm.tqdm(loader)):
        videos, labels, contacts = batch
        videos = videos.to('cuda')
        
        label_dset += [labels.reshape(-1)]
        contacts_dset += [contacts]
        # input is (Bs, T, 3, H, W)
        
        if args.task == 'ocp':
            output = model.extract_features(videos)
        else:
            output = model.extract_features_ocd(videos)

        features_dset += [out]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='physion')
    parser.add_argument('--task', type=str, default='ocp')
    parser.add_argument('--n-past', type=int, required=True)
    parser.add_argument('--n-features', type=int, default=1)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    main(args)

