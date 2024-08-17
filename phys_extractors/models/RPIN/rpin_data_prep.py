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
    #kwargs = {'pin_memory': True, 'num_workers': 8}
    data_loader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=1, 
                                               shuffle=False) 
                                               #**kwargs,)
    
    f = h5py.File(args.save_path, "w")
    dset1 = f.create_dataset("data", (len(dataset), 15, 3, 256, 256), dtype='f')
    dset2 = f.create_dataset("rois", (len(dataset), 25, 10, 4), dtype='f')
    dset3 = f.create_dataset("labels", (len(dataset), 18, 10, 4), dtype='f')
    dset4 = f.create_dataset("data_last", (len(dataset), 7, 3, 256, 256), dtype='f')
    dset5 = f.create_dataset("binary_labels", (len(dataset)), dtype='i')
    dset6 = f.create_dataset("ignore_mask", (len(dataset), 10), dtype='f')  
    dset7 = f.create_dataset("contacts", (len(dataset), 2), dtype='f')
    dset8 = f.create_dataset("collision_ind", (len(dataset)), dtype='f')
    dt = h5py.special_dtype(vlen=str)
    dset9 = f.create_dataset("filenames", (len(dataset)), dtype=dt)

    stimulus_map = {}
    all_scenarios = {'collide': [], 'drop': [], 'support': [], 'link': [], 
                     'roll': [], 'contain': [], 'dominoes': []}
    length = []
    idx = 0
    for i, batch in tqdm.tqdm(enumerate(data_loader)):
        print(i)
        data, rois, labels, data_last, ignore_mask, filename, \
        binary_labels, contacts, collision_ind = batch['data'], batch['rois'], batch['labels'], \
                                            batch['data_last'], batch['ignore_mask'], \
                                            batch['stimulus_name'][0], batch['binary_labels'], \
                                            batch['contacts'], batch['collision_ind']
        
        dset1[i] = data
        dset2[i] = rois
        dset3[i] = labels
        dset4[i] = data_last
        dset5[i] = binary_labels
        dset6[i] = ignore_mask
        dset7[i] = contacts
        dset8[i] = collision_ind
        dset9[i] = filename.decode()
        
        if not filename.decode() in stimulus_map.keys():
            stimulus_map[filename.decode()] = idx
            idx += 1
    
        for key in all_scenarios.keys():
            if key in filename.decode():
                all_scenarios[key].append(i)

    f.close()
    
    

    with open(args.map_dir, 'w') as f:
        import json
        json.dump(stimulus_map, f)
        
    with open(args.indices_scenario_dir, 'w') as f:
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
                        default='test_scenario_map.json',
                        help='Path to save data on')
    parser.add_argument('-i', '--indices-scenario-dir', type=str,
                    default='test_json.json',
                    help='Path to save data on')

    args = parser.parse_args()
    main(args)
