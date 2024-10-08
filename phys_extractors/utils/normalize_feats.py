import argparse
import h5py
import numpy as np
    
from torch import nn
import torch

    
def normalize(args):
    # Load the HDF5 file
    data = h5py.File(args.input)
    features = data['features'+args.spatin][:]
    labels = data['label'][:]
    
    test_data = h5py.File(args.test_input)
    test_features = test_data['features'+args.spatin][:]
    test_labels = test_data['label'][:]
    
    # transformation func
    pool = nn.AdaptiveAvgPool3d((None, None, 1))
    feats = pool(torch.tensor(features))
    test_feats = pool(torch.tensor(test_features))
    
    # Normalize the data
    # Apply same transformation for train and test
    new_feats = (feats - feats.mean(dim=0)) / feats.std(dim=0)
    new_test_feats = (test_feats - feats.mean(dim=0)) / feats.std(dim=0)

    # Save the new values to the same HDF5 file
    with h5py.File(args.output, 'w') as hf:
        hf.create_dataset('features', data=new_feats)
        hf.create_dataset('label', data=labels)
        
    # Save the new values to the same HDF5 file
    with h5py.File(args.test_output, 'w') as hf:
        hf.create_dataset('features', data=new_test_feats)
        hf.create_dataset('label', data=test_labels)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input hdf5 file")
    parser.add_argument("--test_input", type=str, required=True, help="input hdf5 file")
    parser.add_argument("--output", type=str, required=True, help="output hdf5 file")
    parser.add_argument("--test_output", type=str, required=True, help="output hdf5 file")
    parser.add_argument("--spatin", type=str, default='', choices=['_beta', '_gamma'])
    args = parser.parse_args()
    
    normalize(args)