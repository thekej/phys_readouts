from . import DFM_physion
from . import FitVid
from . import R3M
from . import mcvd_pytorch
from . import pixelnerf

import torch


def get_fourier_features(feats):
    '''

    :param feats: [B, H, W, D]
    :return: [B, 16, D]
    '''
    feats = feats.cpu()
    combined_features_uncond = feats

    b, h, w, c = combined_features_uncond.shape

    # compute first second and third moment of features
    feats_flat = combined_features_uncond.flatten(1, 2)
    mean = torch.mean(feats_flat, dim=1, keepdim=True)
    std = torch.std(feats_flat, dim=1, keepdim=True)
    skew = torch.mean((feats_flat - mean) ** 3, dim=1, keepdim=True)
    kurt = torch.mean((feats_flat - mean) ** 4, dim=1, keepdim=True)

    combined_features_uncond = feats_flat.view(b, h, w, c)

    combined_features_uncond = combined_features_uncond.permute(0, 3, 1, 2).flatten(0, 1)

    fft = torch.fft.fft2(combined_features_uncond.to(dtype=torch.float32)).view(b, c, h, w).permute(0, 2, 3, 1) # b, h, w, c

    magnitude = torch.abs(fft)

    k = 6

    # Create a tensor to store the indices of the principal components
    principal_indices = torch.zeros(b, c, k, 2, dtype=torch.long).to(fft.device)

    for i in range(b):
        for j in range(c):
            # Flatten the 2D magnitude and get the top-k indices
            _, indices_1d = torch.topk(magnitude[i, :, :, j].flatten(), k)

            # Convert 1D indices back to 2D indices
            principal_indices[i, j] = torch.stack((torch.div(indices_1d, h, rounding_mode='trunc'), indices_1d % h),
                                                  dim=1)

    principal_components = torch.zeros(b, c, k, dtype=torch.complex64).to(fft.device)

    for i in range(b):
        for j in range(c):
            for l in range(k):
                x, y = principal_indices[i, j, l]
                principal_components[i, j, l] = fft[i, x, y, j]

    # split complex into real and imaginary
    real = principal_components.real
    imag = principal_components.imag

    # concatenate real and imaginary
    real_imag = torch.cat([real, imag], dim=2).permute(0, 2, 1)

    # concatenate real, imaginary, mean, std, skew, kurt
    combined_features_uncond = torch.cat([real_imag, mean, std, skew, kurt], dim=1)

    return combined_features_uncond

import torch.nn.functional as F

class VideoTransform:
    def __init__(self):
        pass

    def resize(self, tensor, size):
        # Resize using F.interpolate
        # This function can handle batched input, so no need to loop
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)

    def center_crop(self, tensor, output_size):
        # Center crop function for tensors for batched input
        th, tw = output_size
        h, w = tensor.shape[-2:]
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return tensor[..., i:i+th, j:j+tw]

    def __call__(self, batch):
        # Assuming batch shape is (bs, T, h, w, C)
        batch = batch.float()  # Ensure the batch is of float32 datatype
        
        # Convert the batch from (bs, T, h, w, C) to (bs*T, C, h, w)
        batch_size, T, h, w, C = batch.shape
        batch = batch.permute(0, 1, 4, 2, 3).reshape(-1, C, h, w)

        # Apply resizing
        batch = self.resize(batch, (256, 256))
        # Apply center cropping
        batch = self.center_crop(batch, (224, 224))

        # Reshape the batch back to (bs, T, C, h', w')
        batch = batch.reshape(batch_size, T, C, 224, 224)
        return batch
