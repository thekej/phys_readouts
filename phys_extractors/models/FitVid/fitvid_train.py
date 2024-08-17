import argparse
import os
import random
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from fitvid_loader import PhysionDataset, Ego4D
from fitvid import FitVid


def validate_model(training_logs, model, batch_idx, total_batches,
                   val_loader, device, epoch, args, loss=None,
                   full=True):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for c, data in enumerate(val_loader):
            if c > 100 and not full:
                break
            # Send data and target to device
            data = data.to(device)
            # Forward pass
            output = model(data)
            step_loss = output['loss'].mean()
            # Calculate loss
            val_loss += step_loss.item()
            # Print results for this epoch
        
        log_sentence = ''
        if batch_idx is not None:
            log_sentence = 'Epoch [{}/{}], Step[{}/{}], Train Loss: {}, Val Loss: {}'.format(epoch+1, args.num_epochs, batch_idx + 1, total_batches, loss.item(), val_loss/len(val_loader))
        else:
            log_sentence = 'Train Epoch [{}/{}] is done, our val Loss: {}'.format(epoch+1, args.num_epochs, val_loss/len(val_loader))
        print(log_sentence)
        training_logs += [log_sentence+'\n']
    
    with open(args.log_files, 'w') as f:                    
        f.writelines(training_logs)

def train(args):
    # Define the device to run the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Load the model
    model = FitVid()
    model = model.to(device)
    
    
    # Load the dataset
    if args.data_type == 'physion':
        # setup train and val indices
        random.seed(4)
        indices = list(range(args.data_size))
        random.shuffle(indices)
        train_indices = range(int(args.data_size * 0.9))
        val_indices = list(set(indices) - set(train_indices))
        train_dataset = PhysionDataset(args.data_path, indices=train_indices)
        val_dataset = PhysionDataset(args.data_path, indices=val_indices)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print('Data Loaded')
    else:

        with open( '/ccn2/u/thekej/ego4d/kinetics_400_train_list.txt', 'r') as f:
            all_training_videos = f.readlines()
        all_training_videos = [_.strip().split(' ')[0] for _ in all_training_videos]

        with open('/ccn2/u/thekej/ego4d/kinetics_400_val_list.txt', 'r') as f:
            all_validation_videos = f.readlines()
        all_validation_videos = [_.strip().split(' ')[0] for _ in all_validation_videos]

        with open('/ccn2/u/thekej/ego4d/ego4d_train_list_320p_chunked.txt', 'r') as f:
            ego4d_videos = f.readlines()
        ego4d_videos = [_.strip().split(' ')[0] for _ in ego4d_videos]
        
        train_dataset = Ego4D(clips=all_training_videos + ego4d_videos,
                                is_color=True,
                                modality='rgb',
                                new_length=25,
                                new_step=2,
                                video_loader=True,
                                use_decord=True)
        
        val_dataset = Ego4D(clips=all_validation_videos,
                            is_color=True,
                            modality='rgb',
                            new_length=25,
                            new_step=2,
                            video_loader=True,
                            use_decord=True)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=96)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=96)
        print('Data Loaded')

    

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Wrap the model with DataParallel to use multiple GPUs
    model = torch.nn.DataParallel(model)

    # Train the model
    training_logs = []
    print('Start training')
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
            #simulated_states = output['simulated_states']
            #observed_states = output['observed_states']
            loss = output['loss'].mean()#nn.MSELoss()(simulated_states, observed_states)
            
            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            if (batch_idx + 1) % args.log_step == 0:
                #if args.data_type == 'physion':
                #    validate_model(training_logs, model, batch_idx, len(train_loader), 
                #                   val_loader, device, epoch, args, loss, full=False)
                #    model.train()
                #else:
                log_sentence = 'Epoch [{}/{}], Step [{}/{}], Loss: {}'.format(epoch+1, args.num_epochs, 
                                                                              batch_idx, len(train_loader), 
                                                                              loss.item())
                print(log_sentence)
                training_logs += [log_sentence+'\n']

                with open(args.log_files, 'w') as f:                    
                    f.writelines(training_logs)
            if (batch_idx + 1) % args.save_step == 0:
                torch.save(model.state_dict(), args.save_path + 'checkpoint_%d_%d.pt'%(epoch, batch_idx))

        # Evaluate on validation set
        validate_model(training_logs, model, None, len(train_loader), val_loader, device, epoch, args, loss=None, full=True)
        torch.save(model.state_dict(), args.save_path + 'checkpoint_%d.pt'%epoch)
    # Save the last model
    torch.save(model.state_dict(), args.save_path + 'checkpoint_final.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a pfR3M_ID model.')
    parser.add_argument('--data-path', type=str, 
                        default='/home/dbear/BBNet/bbnet/models/VideoMAE-main/video_file_lists',
                        help='Path to the R3M dataset')
    parser.add_argument('--data-size', type=int,
                        help='Number of samples in the dataset')
    parser.add_argument('--data-type', type=str, default='ego4d',
                        help='Number of samples in the dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--num-epochs', type=int, default=1,
                        help='Number of epochs to train for')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Save file path')
    parser.add_argument('--log-step', type=int, default=1,
                        help='Log loss frequency')
    parser.add_argument('--save-step', type=int, default=10000,
                        help='Log loss frequency')
    parser.add_argument('--log-files', type=str, required=True,
                        help='Log file path')
    args = parser.parse_args()
    
    train(args)
    
    
'''
# from line 74 to 83 in r3m/r3m/__init__.py
#rep = torch.nn.DataParallel(rep)
r3m_state_dict = remove_language_head(torch.load(modelpath, map_location=torch.device(device))['r3m'])
new_sd = OrderedDict()
for k, v in r3m_state_dict.items():
    if k.startswith("module."):
        name = k[7:]  # remove 'module.' of dataparallel/DDP
    else:
        name = k
    new_sd[name] = v
rep.load_state_dict(new_sd)
'''
                       
                    
