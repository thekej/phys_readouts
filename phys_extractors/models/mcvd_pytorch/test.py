import sys, os
import glob, os
import mediapy as media
import torch
import h5py
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from load_model_from_ckpt import load_model, get_readout_sampler, init_samples
from datasets import get_dataset, data_transform, inverse_data_transform
from datasets.ucf101 import UCF101Dataset
from runners.ncsn_runner import conditioning_fn

from os.path import expanduser
home = expanduser("~")

def get_dataset(data_path, config, video_frames_pred=0, start_at=0):

    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    else:
        tran_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])


    frames_per_sample = 25#config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0) + video_frames_pred
    dataset = UCF101Dataset(data_path, frames_per_sample=frames_per_sample, 
                            image_size=config.data.image_size, train=True, random_time=True,
                            random_horizontal_flip=config.data.random_flip)
    test_dataset = UCF101Dataset(data_path, frames_per_sample=frames_per_sample, 
                                     image_size=config.data.image_size, train=False, random_time=True,
                                     random_horizontal_flip=False, total_videos=256)

    subset_num = getattr(config.data, "subset", -1)
    if subset_num > 0:
        subset_indices = list(range(subset_num))
        dataset = Subset(dataset, subset_indices)

    test_subset_num = getattr(config.data, "test_subset", -1)
    if test_subset_num > 0:
        subset_indices = list(range(test_subset_num))
        test_dataset = Subset(test_dataset, subset_indices)

    return dataset, test_dataset


GENERATION_LENGTH = 20
PRED_LENGTH = 4

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# SET THESE!!!
DATA_PATH = '/ccn2/u/thekej/phys_h5'


EXP_PATH = "/ccn2/u/thekej/ucf10132_big192_288_4c4_unetm_spade/logs/"
ckpt_path = glob.glob(os.path.join(EXP_PATH, "checkpoint_*.pt"))[0]
scorenet, config = load_model(ckpt_path, device)
sampler = get_readout_sampler(config)

print(device)
print(ckpt_path)
print(config)

dataset, test_dataset = get_dataset(DATA_PATH, config, video_frames_pred=config.data.num_frames)

test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False,
                         num_workers=config.data.num_workers, drop_last=True)
test_iter = iter(test_loader)

#f = h5py.File("mytestfile.hdf5", "w")
#dset1 = f.create_dataset("gamma", (32, 288, 64, 64), dtype='f')
#dset2 = f.create_dataset("beta", (32, 288, 64, 64), dtype='f')
#dset3 = f.create_dataset("mid_embed", (32, 1152, 8, 8), dtype='f')


for i, (test_x, _) in enumerate(test_loader):
    print(test_x.shape)
    input_frames = data_transform(config, test_x)
    print(input_frames.shape)
    break

    real, cond, cond_mask = conditioning_fn(config, input_frames, num_frames_pred=config.data.num_frames,
                                        prob_mask_cond=getattr(config.data, 'prob_mask_cond', 0.0),
                                        prob_mask_future=getattr(config.data, 'prob_mask_future', 0.0))
    for j in range(GENERATION_LENGTH // PRED_LENGTH):
        init = init_samples(len(real), config)
        pred, a, e, v = sampler(init, scorenet, cond=cond, cond_mask=cond_mask, subsample=10, verbose=True)
        cond = pred
        print(pred.shape, a.shape, e.shape, v.shape)
    if pred.shape[0] == config.training.batch_size:
        dset1[i * config.training.batch_size: (i+1)*config.training.batch_size, :] = a.cpu().numpy()
        dset2[i * config.training.batch_size: (i+1)*config.training.batch_size, :] = e.cpu().numpy()
        dset3[i * config.training.batch_size: (i+1)*config.training.batch_size, :] = v.cpu().numpy()
    else:
        dset1[i * config.training.batch_size:, :] = a.cpu().numpy()
        dset2[i * config.training.batch_size:, :] = e.cpu().numpy()
        dset3[i * config.training.batch_size:, :] = v.cpu().numpy()
    break
