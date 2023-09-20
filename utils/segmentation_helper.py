import io

import os
import torch.nn

import h5py
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from skimage.transform import resize
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F


def get_image(raw_img, pil=False):
    '''
    raw_img: binary image to be read by PIL
    returns: HxWx3 image
    '''
    img = Image.open(io.BytesIO(raw_img))

    if pil:
        return img

    return np.array(img)


def get_num_frames(h5_file):
    return len(h5_file['frames'].keys())


def index_img(h5_file, index, suffix='', pil=False):
    # print(index)
    if index > len(h5_file['frames']) - 1:
        # set index to last frame
        # print("inside if")
        index = len(h5_file['frames']) - 1

    img0 = h5_file['frames'][str(index).zfill(4)]['images']

    rgb_img = get_image(img0['_img' + suffix][:], pil=pil)

    segments = np.array(get_image(img0['_id' + suffix][:]))

    return rgb_img, segments


def index_imgs(h5_file, indices, static=False, suffix='', pil=False):
    if not static:
        all_imgs = []
        all_segs = []
        for ct, index in enumerate(indices):
            rgb_img, segments = index_img(h5_file, index, suffix=suffix, pil=pil)
            all_imgs.append(rgb_img)
            all_segs.append(segments)
    else:
        rgb_img, segments = index_img(h5_file, indices[0])
        all_imgs = [rgb_img]
        all_segs = [segments]

        for ct, index in enumerate(indices[1:]):
            all_imgs.append(rgb_img)
            all_segs.append(segments)

    if not pil:
        all_imgs = np.stack(all_imgs, 0)
    all_segs = np.stack(all_segs, 0)

    return all_imgs, all_segs


def get_object_masks(seg_imgs, seg_colors, background=True):
    # if len(seg_imgs.shape) == 3:
    #     seg_imgs = np.expand_dims(seg_imgs, 0)
    #     is_batch = False
    # else:
    #     is_batch = True

    obj_masks = []
    for scol in seg_colors:
        mask = (seg_imgs == scol)  # .astype(np.float)

        # If object is not visible in the frame
        # if mask.sum() == 0:
        #     mask[:] = 1
        obj_masks.append(mask)

    obj_masks = np.stack(obj_masks, 0)
    obj_masks = obj_masks.min(axis=-1)  # collapse last channel dim

    if background:
        bg_mask = ~ obj_masks.max(0, keepdims=True)
        obj_masks = np.concatenate([obj_masks, bg_mask], 0)
    obj_masks = np.expand_dims(obj_masks, -3)  # make N x 1 x H x W

    # if not is_batch:
    #     obj_masks = obj_masks[0]
    return obj_masks


class Physion(Dataset):

    def __init__(self, hdf5_path, indices=None):
        with h5py.File(hdf5_path) as f:
            features = f['features'][:]
            filenames = f['filenames'][:]
            masks = f['masks'][:]
        self.filenames = filenames
        self.features = features
        self.masks = masks
        self.indices = indices

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.filenames)

    def __getitem__(self, idx):
        # breakpoint()
        if self.indices is not None:
            idx = self.indices[idx]

        filename = self.filenames[idx]
        feature = self.features[idx]
        mask = self.masks[idx]

        return filename, feature, mask


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of shape [B, N, HW].
                The predictions for each example.
        targets: A float tensor with shape [B, M, HW]. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum("bnc,bmc->bnm", inputs, targets)
    denominator = inputs.sum(-1)[:, :, None] + targets.sum(-1)[:, None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor shape [B, N, HW].
                The predictions for each example.
        targets: A float tensor with shape [B, M, HW]. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[-1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("bnc,bmc->bnm", pos, targets) + torch.einsum("bnc,bmc->bnm", neg, (1 - targets))

    return loss / hw


def batch_hungarian_matcher(cost_matrix):
    """
    Args:
        cost_matrix: a float tensor of shape [B, N, M]

    Returns:
        indices assignment
    """
    B = cost_matrix.shape[0]
    indices = []
    for i in range(B):
        indices.append(np.stack(linear_sum_assignment(cost_matrix[i]), axis=-1))

    match_idx = torch.tensor(np.stack(indices, 0))
    batch_idx = torch.arange(B).view(B, 1, 1).expand(-1, match_idx.shape[1], -1)
    idx = torch.cat([batch_idx, match_idx], dim=-1)
    return idx

# Dataloader
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def batch_iou(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the iou, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of shape [B, N, HW].
                The predictions for each example.
        targets: A float tensor with shape [B, M, HW]. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    numerator = torch.einsum("bnc,bmc->bnm", inputs, targets)
    denominator = ((inputs[:, :, None] + targets[:, None, :]) > 0).sum(-1) + 1e-6
    iou = numerator / denominator
    return iou


    # Save model
def compute_mean_iou_over_dataset(dataloader, model, upsample_size, size, dir_save_images, permute=True):
    mean_iou_list = []
    for i, batch in enumerate(dataloader):
        _, features, masks = batch
        # Features (TODO: Rahul)
        features = resize_tensor(features, size) # [B, T, h, w, Dim]
        
        # Targets
        masks = torch.tensor(masks).float().squeeze(2)
        # breakpoint()
        target = F.interpolate(masks.cuda(), size=size, mode='nearest').flatten(2, 3).squeeze(1)  # [B, 1, hw]

        # Decode
        logit = model(features.cuda()).squeeze(1)  # [B, 1, hw]
        pred = (logit>0).float() #logit.sigmoid().argmax(1)

        assert pred.shape[0] == 1, "only implement for batch size 1"

        iou_cost = batch_iou(pred, target)
        mean_iou_list.append(iou_cost.detach().cpu().numpy())

        # breakpoint()

        if i <= 5:

            batch_idx = 0
            fig, axs = plt.subplots(2, 1, squeeze = False, figsize=(10, 3))
            _iou = mean_iou_list[batch_idx]
            p = pred[batch_idx].view(size[0], size[1]).cpu().detach().bool()
            t = target[batch_idx].view(size[0], size[1]).cpu().detach().bool()
            try:
                axs[0, 0].imshow(p)
            except:
                breakpoint()
            axs[1, 0].imshow(t)
            axs[0, 0].set_title(f'(iou:{_iou:.3f})')
            axs[1, 0].set_title(f'GT')

            #set suptitle as the dice loss and f1 loss

            plt.savefig(os.path.join(dir_save_images, f'{i}.png'))
            plt.close()

    # breakpoint()

    return np.mean(mean_iou_list)


def resize_tensor(input_tensor, new_size):
    # Reorder dimensions to [B, T*C, H, W]
    B, T, H, W, C = input_tensor.shape
    reordered = input_tensor.permute(0, 1, 4, 2, 3).reshape(B, T*C, H, W)

    # Use F.interpolate to resize
    resized = F.interpolate(reordered, size=new_size, mode='bilinear', align_corners=False)
    
    # Reorder dimensions back to [B, T, h, w, C]
    B, _, h, w = resized.shape
    output = resized.reshape(B, T, C, h, w).permute(0, 1, 3, 4, 2)

    return output