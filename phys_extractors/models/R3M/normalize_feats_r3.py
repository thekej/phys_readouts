import argparse
import h5py
import numpy as np
    
from torch import nn
import torch

    
def normalize(args):
    # Load the HDF5 file
    data = h5py.File(args.input)
    observed = data["observed"][:]
    observed_full_outcome = data["observed_full_outcome"][:]
    simulation = data["simulation"][:]
    labels = data['label'][:]
    
    # Concat full scenarios
    complete_ofo = np.concatenate([observed, observed_full_outcome], axis=1)
    simulated_ofo = np.concatenate([observed, simulation], axis=1)
    
    test_data = h5py.File(args.test_input)
    test_observed = test_data["observed"][:]
    test_observed_full_outcome = test_data["observed_full_outcome"][:]
    test_simulation = test_data["simulation"][:]
    test_labels = test_data['label'][:]
    
    # Concat full scenarios
    test_complete_ofo = np.concatenate([test_observed, test_observed_full_outcome], axis=1)
    test_simulated_ofo = np.concatenate([test_observed, test_simulation], axis=1)
    print(test_complete_ofo.shape)
    
    # Normalize the data
    # Apply same transformation for train and test
    new_observed = (observed - np.mean(observed, axis=0)) / np.std(observed, axis=0)
    new_test_observed = (test_observed - np.mean(observed, axis=0)) / np.std(observed, axis=0)
    new_complete_ofo = (complete_ofo - np.mean(complete_ofo, axis=0)) / np.std(complete_ofo, axis=0)
    new_test_complete_ofo = (test_complete_ofo - np.mean(complete_ofo, axis=0)) / np.std(complete_ofo, axis=0)
    new_simulated_ofo = (simulated_ofo - np.mean(simulated_ofo, axis=0)) / np.std(simulated_ofo, axis=0)
    new_test_simulated_ofo = (test_simulated_ofo - np.mean(simulated_ofo, axis=0)) / np.std(simulated_ofo, axis=0)
    
    # Save the new values to the same HDF5 file
    with h5py.File(args.output, 'w') as hf:
        hf.create_dataset('observed', data=new_observed)
        hf.create_dataset('observed_full_outcome', data=new_complete_ofo)
        hf.create_dataset('simulation', data=new_simulated_ofo)
        hf.create_dataset('label', data=labels)
        
    # Save the new values to the same HDF5 file
    with h5py.File(args.test_output, 'w') as hf:
        hf.create_dataset('observed', data=new_test_observed)
        hf.create_dataset('observed_full_outcome', data=new_test_complete_ofo)
        hf.create_dataset('simulation', data=new_test_simulated_ofo)
        hf.create_dataset('label', data=test_labels)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input hdf5 file")
    parser.add_argument("--test_input", type=str, required=True, help="input hdf5 file")
    parser.add_argument("--output", type=str, required=True, help="output hdf5 file")
    parser.add_argument("--test_output", type=str, required=True, help="output hdf5 file")
    args = parser.parse_args()
    
    normalize(args)