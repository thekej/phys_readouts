import argparse
import h5py
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


def objective(trial, args):
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1)
    lr = trial.suggest_float("lr", 1e-8, 1e-1)
    
    #set seed for reproducing experiments
    random.seed(10)
    torch.manual_seed(42)
    
    #Split data
    indices = range(args.data_size)
    train_set = set(random.sample(indices, int(len(indices) * 0.9)))
    val_set = set(indices) - train_set
    
    # Load the data
    train_dataset = readout_feats_loader.FeaturesDataset(args.data_path, list(train_set))
    val_dataset = readout_feats_loader.FeaturesDataset(args.data_path, list(val_set))
    test_dataset = readout_feats_loader.FeaturesDataset(args.test_path, list(val_set))    
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, persistent_workers=True, 
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
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
        auto_lr_find=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=5)
    trainer.fit(model, train_loader, val_loader)
    
    return trainer.callback_metrics['val_accuracy'].item()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model-name', type=str, help='The model class to use')
    parser.add_argument('--data-path', type=str, help='The path to the h5 file')
    parser.add_argument('--test-path', type=str, help='The path to the h5 file')
    parser.add_argument('--data-size', type=int, required=True, help='Dataset size')
    parser.add_argument('--save-path', type=str, help='The path to save the checkpoints')
    parser.add_argument('--n-epochs', type=int, default=150, help='Number of minimum epochs to train (default: 10)')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of gpu devices to train on')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=16, help='Number of workers')
    
    args = parser.parse_args()
    

    # Wrap the objective inside a lambda and call objective inside it
    func = lambda trial: objective(trial, args)

    # Pass func to Optuna studies
    study = optuna.create_study(direction='maximize')
    study.optimize(func, n_trials=100)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    res = {}
    res['Val acc'] = trial.value
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        res[key] = value

    with open(args.save_path, 'w') as f:
        import json 

        json.dump(res, f)
