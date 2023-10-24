import argparse
import numpy as np
import os.path as osp
import os
import tqdm 
import h5py
import torch 

from models import Extractor, UnifiedPhysion
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
    dataset = UnifiedPhysion(args.data_path, frame_duration=args.frame_duration,
                            ocd=args.ocd, video_len = args.video_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    data_size = len(loader)#5608
        
    # set up new dataset
    label_dset = []
    contacts_dset = []
    features_dset = []
    filename = []

    print('start extraction')
    for i, batch in enumerate(tqdm.tqdm(loader)):
        videos, labels, contacts = batch['video'], batch['label'], batch['contacts']
        
        videos = videos.to('cuda')
        
        label_dset += [labels.reshape(-1)]
        contacts_dset += [contacts]
        filename += batch['filename']
        # input is (Bs, T, 3, H, W)
        
        if args.task == 'ocp':
            output = model.extract_features(videos)
        else:
            output = model.extract_features_ocd(videos)

        features_dset += [output]


    with h5py.File(args.save_file, "w") as hf:
        hf.create_dataset("features", data=np.array(features_dset, dtype=float))
        hf.create_dataset("labels", data=np.array(label_dset, dtype=float))
        hf.create_dataset("contacts", data=np.array(contacts_dset, dtype=int))
        dt = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset("stimuli_name", data=np.array(filename, dtype=dt))



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

