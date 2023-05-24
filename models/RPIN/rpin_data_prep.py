import argparse
import io
import glob
import h5py
import numpy as np
import os
import torch
import tqdm

from physion_loader import RPINDataset

from os import listdir
from os.path import isfile, join
from PIL import Image
from torchvision import transforms


def main(args):
    # set up new dataset
    dataset = RPINDataset(args.data_dir)
    kwargs = {'pin_memory': False, 'num_workers': 16}
    data_loader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=1, 
                                               shuffle=False, 
                                               **kwargs,)
    
    f = h5py.File(args.save_path, "w")
    dset1 = f.create_dataset("data", (len(dataset), 15, 3, 256, 256), dtype='f')
    dset2 = f.create_dataset("rois", (len(dataset), 25, 10, 4), dtype='f')
    dset3 = f.create_dataset("labels", (len(dataset), 18, 10, 4), dtype='f')
    dset4 = f.create_dataset("data_last", (len(dataset), 7, 3, 256, 256), dtype='f')
    dset5 = f.create_dataset("binary_labels", (len(dataset)), dtype='i')
    dset6 = f.create_dataset("ignore_mask", (len(dataset), 10), dtype='f')    

    stimulus_map = {}
    length = []
    idx = 0
    for i, batch in tqdm.tqdm(enumerate(data_loader)):
        data, rois, labels, data_last, ignore_mask, stimulus_name, \
        binary_labels = batch['data'], batch['rois'], batch['labels'], batch['data_last'], batch['ignore_mask'], \
                        batch['stimulus_name'][0], batch['binary_labels']
        
        dset1[i] = data
        dset2[i] = rois
        dset3[i] = labels
        dset4[i] = data_last
        dset5[i] = binary_labels
        dset6[i] = ignore_mask
        
        if not stimulus_name in stimulus_map.keys():
            stimulus_map[str(stimulus_name)] = idx
            idx += 1

    f.close()

    with open(args.map_dir, 'w') as f:
        import json
        json.dump(stimulus_map, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str,
                        help='Path to save data on')
    parser.add_argument('-o', '--save-path', type=str,
                        default='encoded_data.',
                        help='Path to save data on')
    parser.add_argument('-m', '--map-dir', type=str,
                        default='encoded_data.',
                        help='Path to save data on')

    args = parser.parse_args()
    main(args)
