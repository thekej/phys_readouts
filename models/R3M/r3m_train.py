import argparse
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from r3m_loader import R3MTrainDataset
from r3m_model import FrozenPretrainedEncoder, load_model, pfR3M_LSTM_physion

def train(args):
    # Define the device to run the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup train and val indices
    indices = range(args.data_size)
    train_indices = range(int(args.data_size * 0.9))
    val_indices = list(set(indices) - set(train_indices))
    
    
    # Load the dataset
    train_dataset = R3MTrainDataset(args.data_path, indices=train_indices)
    val_dataset = R3MTrainDataset(args.data_path, indices=val_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load the model
    model = pfR3M_LSTM_physion(n_past=7)
    model = model.to(device)

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Wrap the model with DataParallel to use multiple GPUs
    model = torch.nn.DataParallel(model)

    # Train the model
    for epoch in range(args.num_epochs):
        # Set model to training mode
        model.train()

        # Train on each batch of data
        for batch_idx, data in enumerate(train_loader):
            # Send data and target to device
            data = data.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate loss
            simulated_states = output['simulated_states']
            observed_states = output['observed_states']
            loss = nn.MSELoss()(simulated_states, observed_states)
            
            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                # Send data and target to device
                data = data.to(device)

                # Forward pass
                output = model(data)
                simulated_states = output['simulated_states']
                observed_states = output['observed_states']
                step_loss = nn.MSELoss()(simulated_states, observed_states)

                # Calculate loss
                val_loss += step_loss.item()

        # Print results for this epoch
        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'
              .format(epoch+1, args.num_epochs, loss.item(), val_loss/len(val_loader)))

    # Save the last model
    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a pfR3M_ID model.')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the R3M dataset')
    parser.add_argument('--data-size', type=int, required=True,
                        help='Number of samples in the dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Save file path')
    args = parser.parse_args()
    
    train(args)
                       
                    