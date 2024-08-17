import argparse
import numpy as np
import os.path as osp
import os
import tqdm 
import h5py
import torch 

from models.feature_extractors import Extractor
from models.unified_loader import UnifiedPhysion
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
    dataset = UnifiedPhysion(args.data_path, 
                             frame_duration=args.frame_duration,
                             ocd= args.task == 'ocd_cwm', 
                             video_len = args.video_len,
                             n_context= args.n_past)
    loader = DataLoader(dataset, batch_size=args.batch_size, 
                        shuffle=False, num_workers=8)
    #print(loader.num_workers)  # This will print "8" in this example

    data_size = len(loader)
        
    # set up new dataset
    label_dset = []
    contacts_dset = []
    features_dset = []
    filename = []
    import time
    print('start extraction')
    for i, batch in enumerate(tqdm.tqdm(loader)):
        videos, labels, contacts = batch['video'], batch['label'], batch['contacts']
        #videos = videos.to('cuda')
        
        label_dset += [labels.reshape(-1)]
        contacts_dset += [contacts]
        filename += [batch['filename']]
        # input is (Bs, T, 3, H, W)
        t = time.time()
        if args.task == 'ocd':
            output = model.extract_features_ocd(videos)
        else:
            output = model.extract_features(videos)
        t1 = time.time()
        print(t1 - t)
        features_dset += [output.cpu().numpy()]
        
    features_dset = np.concatenate(features_dset, axis=0)
    label_dset = np.concatenate(label_dset, axis=0)
    contacts_dset = np.concatenate(contacts_dset, axis=0)
    filename = np.concatenate(filename, axis=0)

    with h5py.File(args.save_file, "w") as hf:
        hf.create_dataset("features", data=features_dset)
        hf.create_dataset("labels", data=label_dset)
        hf.create_dataset("contacts", data=contacts_dset)
        #hf.create_dataset("stimuli_name", data=filename)
        dt = h5py.special_dtype(vlen=str)  # This specifies a variable-length string
        dset = hf.create_dataset("stimuli_name", (len(filename),), dtype=dt)
        dset[:] = filename



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='physion')
    parser.add_argument('--task', type=str, default='ocp')
    parser.add_argument('--n_past', type=int, required=True)
    parser.add_argument('--n_features', type=int, default=13)
    parser.add_argument('--frame_duration', type=int, default=60)
    parser.add_argument('--video_len', type=int, default=25)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    main(args)

