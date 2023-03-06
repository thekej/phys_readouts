import json
import argparse
import torch
import h5py

import readout_feats_loader

def load_hdf5(file_path, dataset_name, indices=None):
    with h5py.File(file_path, 'r') as file:
        if indices is None:
            dataset = file[dataset_name][...]
        else:
            dataset = file[dataset_name][sorted(indices)]
    return dataset

def map_(args):
    with open(args.map_path, 'r') as f:
        scenarios = json.load(f)

    scenarios_balanced = {'coll' : [], 'domino': [], 'link': [], 
                 'towers': [], 'contain': [], 'drop': [], 
                 'roll': []}

    for s in scenarios.keys():
        
        labels = load_hdf5(args.data_path, 'label', scenarios[s])
        class_counts = torch.bincount(torch.tensor(labels).int())
        min_class = min(class_counts).item()
        print(class_counts)
        print(min_class)
        
        indices = []
        count = [0,0]
        for idx, i in enumerate(labels):
            if count[int(i)] < min_class:
                
                indices += [sorted(scenarios[s])[idx]]
                count[int(i)] += 1
        print('For class %s, we have %d negative and %d positive'%(s, count[0], count[1]))
        scenarios_balanced[s] = indices
        #print(indices)

    with open(args.save_scenario, 'w') as f:
        json.dump(scenarios_balanced, f)
        
    temp = scenarios_balanced.values()
    all_indices = []
    for ind in temp:
        all_indices += ind
    print(len(all_indices))
    with open(args.save_path, 'w') as f:
        json.dump(all_indices, f)
     


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-path', type=str, default=None,
                        help='Path for indices map file')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path for features file')
    parser.add_argument('--data-type', type=str, default=None,
                        help='r3m | mcvd | teco')
    parser.add_argument('--save-scenario', type=str, default=None,
                        help='Save path for scenario dict')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Save path for balanced train set')
    args = parser.parse_args()
    map_(args)