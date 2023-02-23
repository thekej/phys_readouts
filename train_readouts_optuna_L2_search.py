import argparse
import h5py
import json
import optuna
import random
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import Dataset, DataLoader

from utils import readout_feats_loader
from utils.train_utils import get_model

def get_data(args):
    with open(args.balanced_indices, 'r') as f:
        indices = json.load(f)
        
    # account for all but one train protocol
    if args.all_but_one is not None:
        with open(args.train_scenario_map, 'r') as f:
            scenarios_indices = json.load(f)
            banned_scenario = scenarios_indices[args.all_but_one]
            indices = list(set(indices) - set(banned_scenario))
            print('Removing %d datapoints from the %s scenario'%(len(banned_scenario),
                                                                 args.all_but_one))
            
    #split data
    train_set = set(random.sample(indices, int(len(indices) * 0.9)))
    val_set = set(indices) - train_set
    
    if args.data_type == 'mcvd':
        train_dataset = readout_feats_loader.FeaturesDataset(args.data_path, list(train_set), scenario=args.scenario)
        val_dataset = readout_feats_loader.FeaturesDataset(args.data_path, list(val_set), scenario=args.scenario)

        if args.all_but_one is not None:
            with open(args.test_scenario_map, 'r') as f:
                scenarios_indices = json.load(f)
                test_dataset = readout_feats_loader.FeaturesDataset(args.test_path, 
                                                                    scenarios_indices[args.all_but_one],
                                                                    scenario=args.scenario)
        else:
            test_dataset = readout_feats_loader.FeaturesDataset(args.test_path, scenario=args.scenario)
    elif args.data_type == 'r3m':
        train_dataset = readout_feats_loader.R3MFeaturesDataset(args.data_path, list(train_set), scenario=args.scenario)
        val_dataset = readout_feats_loader.R3MFeaturesDataset(args.data_path, list(val_set), scenario=args.scenario)

        if args.all_but_one is not None:
            with open(args.test_scenario_map, 'r') as f:
                scenarios_indices = json.load(f)
                test_dataset = readout_feats_loader.R3MFeaturesDataset(args.test_path, 
                                                                    scenarios_indices[args.all_but_one],
                                                                    scenario=args.scenario)
        else:
            test_dataset = readout_feats_loader.R3MFeaturesDataset(args.test_path, scenario=args.scenario)
        
    return train_dataset, val_dataset, test_dataset

def train(args, lr):
    weight_decay = 0#trial.suggest_float("weight_decay", 1e-6, 1e-1)
    
    #set seed for reproducing experiments
    random.seed(0)
    torch.manual_seed(0)
    
    print('load_data')
    train_dataset, val_dataset, test_dataset = get_data(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=True, 
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, persistent_workers=True, 
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, persistent_workers=True, 
                            pin_memory=True)

    # Load the model
    inputs, labels = next(iter(train_loader))
    input_shape = inputs.shape
    model = get_model(args.model_name, input_shape, weight_decay, lr)
    
    trainer = Trainer(
        devices=args.num_gpus,
        accelerator="auto",
        min_epochs=args.n_epochs,
        max_epochs=args.n_epochs,
        default_root_dir=args.save_path,
        log_every_n_steps=10,
        check_val_every_n_epoch= 1)
    trainer.fit(model, train_loader, val_loader)
        
    trainer.test(dataloaders=test_loader, ckpt_path="best")

def objective(trial, args):
    weight_decay = 0#trial.suggest_float("weight_decay", 1e-6, 1e-1)
    lr = trial.suggest_float("lr", 1e-6, 1e-3)
    
    #set seed for reproducing experiments
    random.seed(0)
    torch.manual_seed(0)
    
    print('load_data')
    train_dataset, val_dataset, test_dataset = get_data(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=True, 
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, persistent_workers=True, 
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, persistent_workers=True, 
                            pin_memory=True)

    # Load the model
    inputs, labels = next(iter(train_loader))
    input_shape = inputs.shape
    model = get_model(args.model_name, input_shape, weight_decay, lr)
    
    trainer = Trainer(
        devices=args.num_gpus,
        accelerator="auto",
        min_epochs=args.n_epochs,
        max_epochs=args.n_epochs,
        default_root_dir=args.save_path,
        log_every_n_steps=10,
        check_val_every_n_epoch= 1)
    trainer.fit(model, train_loader, val_loader)
    
    val_loss = trainer.callback_metrics['val_loss'].item()
    
    #trainer.test(dataloaders=test_loader, ckpt_path="best")
    
    return val_loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--data-path', type=str, help='The path to the h5 file')
    parser.add_argument('--data-type', type=str, choices=['mcvd', 'r3m'])
    parser.add_argument('--test-path', type=str, help='The path to the test file')
    parser.add_argument('--save-path', type=str, help='The path to save the checkpoints')
    parser.add_argument('--scenario', type=str, default='complete')
    parser.add_argument('--all-but-one', type=str, default=None,
                        choices=['collision', 'domino', 'link', 'towers', 
                                 'contain', 'drop', 'roll'],
                        help='in case of all-but-one scenario')
    parser.add_argument('--balanced-indices', type=str, required=True, 
                        help='path for scenario mapping')
    parser.add_argument('--train-scenario-map', type=str, required=True, 
                        help='path for scenario mapping')
    parser.add_argument('--test-scenario-map', type=str, required=True, 
                        help='path for scenario mapping')
    # Acceleration params
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of gpu devices to train on')
    parser.add_argument('--num-workers', type=int, default=96, help='Number of workers')
    # Train params
    parser.add_argument('--model-name', type=str, help='The model class to use')
    parser.add_argument('--n-epochs', type=int, default=50, help='Number of minimum epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=8096, help='Batch size')
    
    args = parser.parse_args()
    

    # Wrap the objective inside a lambda and call objective inside it
    func = lambda trial: objective(trial, args)

    # Pass func to Optuna studies
    #sampler = optuna.samplers.CmaEsSampler(
    #    use_separable_cma=True
    #)
    study = optuna.create_study(direction='minimize')#sampler=sampler, 
    
    print(f"Sampler is {study.sampler.__class__.__name__}")
    study.optimize(func, n_trials=100)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    res = {}
    res['Val loss'] = trial.value
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        res[key] = value
        
    train(args, res['lr'])
