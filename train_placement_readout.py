import argparse
import os
import time
import numpy as np
import torch
from utils.segmentation_helper import *
    
    
class ReadoutModel(torch.nn.Module):
    def __init__(self, feature_dim, num_predicted_masks, kernel_size):
        super(ReadoutModel, self).__init__()

        # Add a 3D Conv to merge the time dimension
        self.time_conv = torch.nn.Conv3d(1, 1, kernel_size=(feature_dim, 1, 1), 
                                         stride=1, padding=0) # Assumes dim is same as time length

        self.upconv = torch.nn.ConvTranspose2d(feature_dim, 128, kernel_size=kernel_size, 
                                               stride=kernel_size, padding=0)

        decoder_layers = [torch.nn.ReLU()]
        decoder_layers.append(torch.nn.Linear(128, num_predicted_masks))

        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, feature):
        feature = feature.float()

        # Apply the time convolution
        # Assuming the input feature is of shape (batch, time, h, w, dim)
        feature = feature.permute(0, 4, 1, 2, 3) # Convert to (batch, dim, time, h, w) for Conv3D
        feature = self.time_conv(feature) 
        feature = feature.squeeze(2) # Remove the time dimension

        feature = self.upconv(feature).permute(0, 2, 3, 1)
        logit = self.decoder(feature)

        # Flattening to get H*W shape for the output
        logit = logit.view(logit.size(0), -1)

        return logit


class SegmentationTrainer:
    def __init__(self, args):
        self.args = args
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        self._setup_paths()
        self._load_data()
        self._initialize_hyperparameters()

    def _setup_paths(self):
        dir_save_weights = '/'.join(self.args.train_features_hdf5.split('/')[:-1]) + '/weights/'
        if not os.path.exists(dir_save_weights):
            os.makedirs(dir_save_weights)

        self.model_save_path = os.path.join(dir_save_weights, 'linear_decoder.pt')

        dir_save_images_val = '/'.join(self.args.train_features_hdf5.split('/')[:-1]) + '/val_images/'
        if not os.path.exists(dir_save_images_val):
            os.makedirs(dir_save_images_val)

        dir_save_images_test = '/'.join(self.args.train_features_hdf5.split('/')[:-1]) + '/test_images/'
        if not os.path.exists(dir_save_images_test):
            os.makedirs(dir_save_images_test)

    def _load_data(self):
        times = [0]
        # Train dataset
        self.physion_train_datset = Physion(gaps=times, hdf5_path=self.args.train_features_hdf5, 
                                            full_video=True, start_zero=False, phase='train')
        self.physion_val_datset = Physion(gaps=times, hdf5_path=self.args.train_features_hdf5, 
                                          full_video=True, start_zero=False, phase='val')
        self.physion_test_datset = Physion(gaps=times, hdf5_path=self.args.test_features_hdf5, 
                                           full_video=True, start_zero=False, phase='test')

        self.train_dataloader = MultiEpochsDataLoader(self.physion_train_datset, batch_size=32, shuffle=True, num_workers=32)
        self.val_loader = MultiEpochsDataLoader(self.physion_val_datset, batch_size=1, shuffle=False, num_workers=1)
        self.test_loader = MultiEpochsDataLoader(self.physion_test_datset, batch_size=1, shuffle=False, num_workers=1)

        h_, w_ = self.physion_train_datset[0]['feature'].shape[:2]
        self.h_ = h_
        self.w_ = w_

    def _initialize_hyperparameters(self):
        self.num_predicted_masks = 11
        self.hidden_dim = 64
        self.feature_dim = self.physion_train_datset.all_features[0].shape[-1]
        self.val_after = 5
        self.lr_list = [1e-2, 1e-3, 1e-4]
        self.num_epochs = 300
        self.size = (256, 256)  # Always power of 2
        self.upsample_size = int(2**np.round(np.log2(self.h_)))
        self.upsample_size = (self.upsample_size, self.upsample_size)
        self.kernel_size = int(self.size[0] // self.upsample_size[0])
        self.convergence_thresh = 4
        self.overall_best_iou = 0
        self.best_lr = self.lr_list[0]

    def train(self):
        for lr in self.lr_list:
            model = ReadoutModel(self.feature_dim, self.num_predicted_masks, self.kernel_size).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            counter_converge = 0
            best_val_iou = 0

            for epoch_num in range(self.num_epochs):
                t = time.time()
                for i, batch in enumerate(self.train_dataloader):
                    _, features, masks = batch
                    # Features
                    features = resize_tensor(features, size)
                    
                    # Targets
                    masks = torch.tensor(masks).float()
                    target = F.interpolate(masks.cuda(), size=size, mode='nearest').squeeze(1).flatten(2, 3)  # [B, M, hw]

                    # Forward pass through the decoder
                    logit = model(feature).squeeze(1)    # [B, 1, hw]

                    # breakpoint()
                    dice_cost = batch_dice_loss(logit, target)
                    f1_cost = batch_sigmoid_ce_loss(logit, target)
                    loss = dice_cost + f1_cost  # [B, N, W]

                    # Optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                print('time:', time.time() - t, f'Epoch:{epoch_num}, Train loss:{loss.item():.5f}')
                # [The validation logic remains the same.]

    def evaluate(self):
        model = ReadoutModel(self.feature_dim, self.num_predicted_masks, self.kernel_size).cuda()
        model.load_state_dict(torch.load(self.model_save_path))
        test_iou = compute_mean_iou_over_dataset(self.test_loader, model, 
                                                 self.upsample_size, self.size, 
                                                 self.dir_save_images_test, permute=False)
        return test_iou

    def save_results(self, test_iou):
        with open(self.args.model_type + '_.txt', 'w') as fileo:
            fileo.write('Test Mean IoU:' + str(test_iou))
            fileo.write('Best lr:' + str(self.best_lr))


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate a logistic regression model')
    # [All the argparse statements go here.]

    args = parser.parse_args()

    trainer = SegmentationTrainer(args)
    trainer.train()
    test_iou = trainer.evaluate()
    trainer.save_results(test_iou)


if __name__ == '__main__':
    main()
