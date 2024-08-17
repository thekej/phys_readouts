import sys, os
import glob, os
import mediapy as media
import torch
import h5py
from torch.utils.data import DataLoader

from models.mcvd_pytorch.load_model_from_ckpt import load_model, get_readout_sampler, init_samples
from models.mcvd_pytorch.datasets import get_dataset, data_transform, inverse_data_transform
from models.mcvd_pytorch.runners.ncsn_runner import conditioning_fn

from os.path import expanduser
home = expanduser("~")

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

f = h5py.File("mytestfile.hdf5", "w")
dset1 = f.create_dataset("gamma", (32, 288, 64, 64), dtype='f')
dset2 = f.create_dataset("beta", (32, 288, 64, 64), dtype='f')
dset3 = f.create_dataset("mid_embed", (32, 1152, 8, 8), dtype='f')


for i, (test_x, _) in enumerate(test_loader):
    input_frames = data_transform(config, test_x)

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
