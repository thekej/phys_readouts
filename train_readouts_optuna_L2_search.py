import argparse
import h5py
import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import Dataset, DataLoader

from utils import readout_feats_loader
from utils.train_utils import get_model


def objective(trial, args):
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-1)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    
   # Load the data
    dataset = readout_feats_loader.FeaturesDataset(args.data_path)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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
        callbacks=[EarlyStopping(monitor="val_acc", mode="max")]
    )
    trainer.fit(model, train_loader, val_loader)
    return trainer.callback_metrics['val_acc'][-1]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model-name', type=str, help='The model class to use')
    parser.add_argument('--data-path', type=str, help='The path to the h5 file')
    parser.add_argument('--save-path', type=str, help='The path to save the checkpoints')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of minimum epochs to train (default: 10)')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of gpu devices to train on')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=32, help='Number of workers')
    
    args = parser.parse_args()
    

    # Wrap the objective inside a lambda and call objective inside it
    func = lambda trial: objective(trial, args)

    # Pass func to Optuna studies
    study = optuna.create_study(direction='minimize')
    study.optimize(func, n_trials=10)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
