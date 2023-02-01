import argparse
import h5py
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import Dataset, DataLoader

from utils import readout_feats_loader
from utils.train_utils import get_model


def train(args):
    torch.set_float32_matmul_precision('medium')
    # Load the data
    dataset = readout_feats_loader.FeaturesDataset(args.data_path)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load the model
    inputs, labels = next(iter(train_loader))
    input_shape = inputs.shape
    model = get_model(args.model_name, input_shape)
    
    trainer = Trainer(
        devices=args.num_gpus,
        accelerator="auto",
        min_epochs=args.n_epochs,
        max_epochs=args.n_epochs,
        default_root_dir=args.save_path,
        auto_lr_find=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")]
    )
    trainer.fit(model, train_loader, val_loader)
    
    trainer.test(dataloaders=val_loader, ckpt_path="best")

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
    
    train(args)
