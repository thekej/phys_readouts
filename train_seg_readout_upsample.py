
import argparse
#create parse args
import numpy as np

parser = argparse.ArgumentParser(description='Train and evaluate a logistic regression model')

#args for path to train features hdf5 file
parser.add_argument('--train_features_hdf5', type=str, default='/ccn2/u/rmvenkat/data/test_with_keypoint_model_3_feats/M4/train_features.hdf5')

#args for path to test features hdf5 file
parser.add_argument('--test_features_hdf5', type=str, default='/ccn2/u/rmvenkat/data/test_with_keypoint_model_3_feats/M4/test_features.hdf5')


#model type
parser.add_argument('--model_type', type=str, default='CWM')

#gpu id
parser.add_argument('--gpu_id', type=int, default=0)

args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

import time

import torch

from seg_utils import *


dir_save_images_val = '/'.join(args.train_features_hdf5.split('/')[:-1]) + '/val_images/'

if not os.path.exists(dir_save_images_val):
    os.makedirs(dir_save_images_val)

dir_save_images_test = '/'.join(args.train_features_hdf5.split('/')[:-1]) + '/test_images/'

if not os.path.exists(dir_save_images_test):
    os.makedirs(dir_save_images_test)



#command to run this script
#python train_seg_readout.py --train_features_hdf5 /ccn2/u/rmvenkat/data/test_with_keypoint_model_3_feats/M4/train_features.hdf5 --test_features_hdf5 /ccn2/u/rmvenkat/data/test_with_keypoint_model_3_feats/M4/test_features.hdf5 --feature_dim 256 --model_type CWM

times = [0]
#train dataset
physion_train_datset = Physion(gaps=times, \
                         hdf5_path=args.train_features_hdf5,
                         full_video=True, start_zero=False, phase='train')

physion_val_datset = Physion(gaps=times, \
                         hdf5_path=args.train_features_hdf5,
                         full_video=True, start_zero=False, phase='val')

physion_test_datset = Physion(gaps=times, \
                         hdf5_path=args.test_features_hdf5,
                         full_video=True, start_zero=False, phase='test')

train_dataloader = MultiEpochsDataLoader(physion_train_datset, batch_size=32, shuffle=True, num_workers=32)

val_loader = MultiEpochsDataLoader(physion_val_datset, batch_size=1, shuffle=False, num_workers=1)

test_loader = MultiEpochsDataLoader(physion_test_datset, batch_size=1, shuffle=False, num_workers=1)

# breakpoint()
h_, w_ = physion_train_datset[0]['feature'].shape[:2]

# Define the decoder MLP
num_predicted_masks = 11
num_hidden_layers = 1
hidden_dim = 64
feature_dim = physion_train_datset.all_features[0].shape[-1]
val_after = 1

lr_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

num_epochs = 1000

size = (256, 256) # always power of 2

upsample_size = int(2**np.round(np.log2(h_)))
upsample_size = (upsample_size, upsample_size)

kernel_size = int(size[0] // upsample_size[0])

convergence_thresh = 20

overall_best_iou = 0

best_lr = lr_list[0]

class ReadoutModel(torch.nn.Module):
    def __init__(self, feature_dim, num_predicted_masks, kernel_size):
        super(ReadoutModel, self).__init__()

        self.upconv = torch.nn.ConvTranspose2d(feature_dim, 128, kernel_size=kernel_size, stride=kernel_size, padding=0)

        decoder_layers = [torch.nn.ReLU()]

        decoder_layers.append(torch.nn.Linear(128, num_predicted_masks))

        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, feature):
        feature = feature.float()
        feature = self.upconv(feature).permute(0, 2, 3, 1)
        logit = self.decoder(feature)
        return logit

for lr in lr_list:

    model = ReadoutModel(feature_dim, num_predicted_masks, kernel_size).cuda()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Num epochs

    counter_converge = 0

    best_val_iou = 0

    for epoch_num in range(num_epochs):
        t = time.time()
        for i, batch in enumerate(train_dataloader):
            # Features
            seg_color = torch.tensor(batch['feature']).squeeze(1)  # / 255. # [B, H, W, 3]
            feature = F.interpolate(seg_color.permute(0, 3, 1, 2).cuda(), upsample_size, mode='bilinear') # [B, 3, h, w]

            # Targets
            obj_mask = torch.tensor(batch['obj_masks']).float().squeeze(2)
            target = F.interpolate(obj_mask.cuda(), size=size, mode='nearest').flatten(2, 3)  # [B, M, hw]

            # Forward pass through the decoder
            logit = model(feature).permute(0, 3, 1, 2).flatten(2, 3)  # [B, N, hw]

            # breakpoint()

            # Compute pairwise cost
            valid = target.sum(-1)[:, None].expand(-1, logit.shape[1], -1) == 0
            # breakpoint()
            dice_cost = batch_dice_loss(logit, target)
            f1_cost = batch_sigmoid_ce_loss(logit, target)
            cost = dice_cost + f1_cost  # [B, N, W]
            cost[valid] = 1e6

            # Hungarian matching
            match_idx = batch_hungarian_matcher(cost.cpu().detach()).permute(2, 0, 1)  # [3, B, N]
            loss_list = loss = cost[list(match_idx)]
            valid = loss_list < 1e6

            # Compute loss value after matching
            loss = (loss * valid).sum() / (valid.sum() + 1e-6)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('time:', time.time() - t, f'Epoch:{epoch_num}, Train loss:{loss.item():.5f}')

        if epoch_num % val_after == 0:
            mean_iou = compute_mean_iou_over_dataset(val_loader, model, upsample_size, size, dir_save_images_val, permute=False)
            print(f'Epoch:{epoch_num}, Val Mean IoU:{mean_iou:.5f}', 'loss:', loss.item())
            if mean_iou > best_val_iou:
                counter_converge = 0
                flag = True
                best_val_iou = mean_iou
                if mean_iou > overall_best_iou:
                    best_lr = lr
                    overall_best_iou = mean_iou
                    print(f'Saving best model with val mean IoU:{mean_iou:.5f}')
                    torch.save(model.state_dict(), 'linear_decoder.pt')
            else:
                counter_converge += 1

        if counter_converge >= convergence_thresh:
            break

model.load_state_dict(torch.load('linear_decoder.pt'))

test_iou = compute_mean_iou_over_dataset(test_loader, model, upsample_size, size, dir_save_images_test, permute=False)

print(f'Test Mean IoU:{test_iou:.5f}')

fileo = open(args.model_type + '_.txt', 'w')
fileo.write('Test Mean IoU:' + str(test_iou))
#best lr
fileo.write('Best lr:' + str(best_lr))
fileo.close()