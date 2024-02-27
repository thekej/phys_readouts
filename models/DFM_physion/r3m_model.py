import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from collections import OrderedDict
from einops import rearrange
from torch.functional import F


R3M_VAL_TRANSFORMS = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
]

def load_model(
    model, model_path, state_dict_key="state_dict"
):
    params = torch.load(model_path, map_location="cpu")
    sd = params
    aran = False
    if model_path == '/ccn2/u/thekej/R3M/ego4d+physionv1/checkpoint.pt':
        aran = True
        sd = sd['state_dict']
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith("module.") and 'r3m' in k and aran:
            name = 'encoder.r3m.' + k[19:]
        elif k.startswith("module.") and 'r3m' in k and not aran:
            name = 'encoder.r3m.module.' + k[19:]  # remove 'module.' of dataparallel/DDP
        elif k.startswith("module.") and 'dynamics' in k:
            name = k[7:]
        elif k.startswith("module."):
            name = k[7:]
        else:
            name = k
        new_sd[name] = v
    model.load_state_dict(new_sd)
    print(f"Loaded parameters from {model_path}")

    # Set model to eval mode'''
    model.eval()

    return model

class ID(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, x):
        assert isinstance(x, list)
        return x[-1]  # just return last embedding


class LSTM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(self.latent_dim, 1024)
        self.regressor = nn.Linear(1024, self.latent_dim)

    def forward(self, x):
        feats = torch.stack(x)  # (T, Bs, self.latent_dim)
        assert feats.ndim == 3
        # note: for lstms, hidden is the last timestep output
        _, hidden = self.lstm(feats)
        # assumes n_layers=1
        x = torch.squeeze(hidden[0].permute(1, 0, 2), dim=1)
        x = self.regressor(x)
        return x


class ResNet50(nn.Module):
    def __init__(self):

        super().__init__()
        from transformers import ResNetModel
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.latent_dim = 2048


    def forward(self, images):
        '''
        images: [B, C, H, W], Image is normalized with imagenet norm
        '''
        input_dict = {'pixel_values': images}

        decoder_outputs = self.model(**input_dict, output_hidden_states=True)

        features = decoder_outputs.last_hidden_state
        
        features = features.reshape(features.shape[0], -1)
        
        features = nn.AdaptiveAvgPool1d(2048)(features.float())

        return features


class DFM(nn.Module):
    def __init__(self):

        super().__init__()

        render_settings = {
            "n_coarse": 64,
            "n_fine": 64,
            "n_coarse_coarse": 32,
            "n_coarse_fine": 0,
            "num_pixels": 64 ** 2,
            "n_feats_out": 64,
            "num_context": 1,
            "sampling": "patch",
            "cnn_refine": False,
            "self_condition": False,
            "lindisp": False,
            # "cnn_refine": True,
        }
        from PixelNeRF import PixelNeRFModelCond

        self.model = PixelNeRFModelCond(
            near=1.0,
            # dataset.z_near, we set this to be slightly larger than the one we used for training to avoid floaters
            far=2,
            model='dit',
            use_first_pool=False,
            mode='cond',
            feats_cond=True,
            use_high_res_feats=True,
            render_settings=render_settings,
            use_viewdir=False,
            image_size=128,
            use_abs_pose=False,
        )
        weights_path = '/ccn2/u/rmvenkat/data/dfm_weights/re10k_model.pt'
        ckpt = torch.load(weights_path, map_location='cpu')

        new_dict = {}
        for key in ckpt['model'].keys():
            if 'enc' in key:
                new_dict[key[6:]] = ckpt['model'][key]

        self.model.load_state_dict(new_dict, strict=False)
        self.latent_dim = 8192

    def forward(self, videos):
        '''
        videos: [B, T, C, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''
        ctxt_rgb = videos.unsqueeze(1)

        b, num_context, c, h, w = ctxt_rgb.shape

        ctxt_rgb = rearrange(ctxt_rgb, "b t h w c -> (b t) h w c")

        t = torch.zeros((b, num_context), device=ctxt_rgb.device, dtype=torch.long)

        t_resnet = rearrange(t, "b t -> (b t)")
        ctxt_inp = ctxt_rgb
        feature_map = self.model.get_feats(ctxt_inp, t_resnet, abs_camera_poses=None)

        # To downscale it by a factor of 4, we are reducing the size of H and W
        # Calculate the new dimensions
        H_new = feature_map.shape[-1] // 4
        W_new = feature_map.shape[-2] // 4

        # Now, we will use the interpolate function from the torch.nn.functional module
        feature_map = F.interpolate(feature_map, size=(H_new, W_new), mode='bilinear', align_corners=False)

        features = feature_map.reshape(feature_map.shape[0], -1)

        features = nn.AdaptiveAvgPool1d(8192)(features.float())
        return features




# Given sequence of images, predicts next latent
class FrozenPretrainedEncoder(nn.Module):
    def __init__(self, encoder, dynamics, n_past=7, full_rollout=False):
        super().__init__()

        self.full_rollout = full_rollout
        self.n_past = n_past
        self.encoder_name = encoder.lower()
        self.dynamics_name = dynamics.lower()
        Encoder = _get_encoder(self.encoder_name)
        self.encoder = Encoder()

        Dynamics = _get_dynamics(self.dynamics_name)
        dynamics_kwargs = {"latent_dim": self.encoder.latent_dim}
        if self.dynamics_name == "mlp":
            dynamics_kwargs["n_past"] = self.n_past
        self.dynamics = Dynamics(**dynamics_kwargs)

    def forward(self, x, n_past=None):
        # set frozen pretrained encoder to eval mode
        self.encoder.eval()
        # x is (Bs, T, 3, H, W)
        assert len(x.shape) == 5
        if x.shape[1] <= self.n_past:
            self.n_past = -1
        
        inputs = x[:, : self.n_past]
        input_states = self.get_encoder_feats(inputs)

        if self.full_rollout:
            # roll out the entire trajectory
            label_images = x[:, self.n_past :]
            rollout_steps = label_images.shape[1]
        else:
            label_images = x[:, self.n_past]
            rollout_steps = 1

        observed_states = self.get_encoder_feats(label_images)
        simulated_states = []
        prev_states = input_states
        for step in range(rollout_steps):
            # dynamics model predicts next latent from past latents
            pred_state = self.dynamics(prev_states)
            simulated_states.append(pred_state)
            # add most recent pred and delete oldest (to maintain a temporal window of length n_past)
            prev_states.append(pred_state)
            prev_states.pop(0)

        input_states = torch.stack(input_states, axis=1)
        observed_states = torch.stack(observed_states, axis=1)
        simulated_states = torch.stack(simulated_states, axis=1)
        assert observed_states.shape == simulated_states.shape

        output = {
            "input_states": input_states,
            "observed_states": observed_states,
            "simulated_states": simulated_states,
        }
        if self.full_rollout:
            output["states"] = torch.cat([input_states, simulated_states], axis=1)
            # adding this one as a visualizable sanity check of feature extractor
            output["inputs_test"] = torch.cat([inputs, label_images], axis=1)
            assert output["inputs_test"].shape == x.shape
            assert np.array_equal(output["inputs_test"].cpu().numpy(), x.cpu().numpy())
            # should be matched in B and T dimensions
            assert output["states"].shape[0] == output["inputs_test"].shape[0]
            assert output["states"].shape[1] == output["inputs_test"].shape[1]
            assert output["states"].ndim >= 3
        return output

    def get_encoder_feats(self, x):
        # applies encoder to each image in x: (Bs, T, 3, H, W) or (Bs, 3, H, W)
        with torch.no_grad():  # TODO: best place to put this?
            if x.ndim == 4:  # (Bs, 3, H, W)
                feats = [self._extract_feats(x)]
            else:
                assert x.ndim == 5, "Expected input to be of shape (Bs, T, 3, H, W)"
                feats = []
                for _x in torch.split(x, 1, dim=1):
                    _x = torch.squeeze(
                        _x, dim=1
                    )  # _x is shape (Bs, 1, 3, H, W) => (Bs, 3, H, W) TODO: put this in _extract_feats?
                    feats.append(self._extract_feats(_x))
        return feats

    def _extract_feats(self, x):
        feats = self.encoder(x)
        feats = torch.flatten(feats, start_dim=1)  # (Bs, -1)
        return feats


# ---Utils---
def _get_encoder(encoder):
    if encoder == "dfm":
        return DFM
    elif encoder == "resnet":
        return ResNet50
    else:
        raise NotImplementedError(encoder)


def _get_dynamics(dynamics):
    if dynamics == "id":
        return ID
    elif dynamics == "lstm":
        return LSTM
    else:
        raise NotImplementedError(dynamics)


def pfDFM_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="dfm", dynamics="lstm", n_past=n_past, **kwargs
    )

