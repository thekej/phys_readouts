import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from collections import OrderedDict

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

class R3M_pretrained(nn.Module):
    def __init__(self):
        from .r3m import load_r3m
        super().__init__()
        self.r3m = load_r3m("resnet50")
        self.latent_dim = 2048  # resnet50 final fc in_features

    def forward(self, x):
        return self.r3m(x * 255.0)  # R3M expects image input to be [0-255]

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


class DINOV2(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel as automodel
        self.model = automodel.from_pretrained('facebook/dinov2-large')
        self.latent_dim = 8192

    def forward(self, images):
        '''
        images: [B, C, H, W], Image is normalized with imagenet norm
        '''
        input_dict = {'pixel_values': images}

        decoder_outputs = self.model(**input_dict, output_hidden_states=True)

        features = decoder_outputs.last_hidden_state
        features_1 = features[:, 0]
        features_2 = nn.AdaptiveAvgPool1d((7168))(features[:, 1:].reshape(features.shape[0], -1).float())
        features = torch.cat((features_1, features_2), dim=1)
        return features


    def extract_features(self, videos):
        '''
        videos: [B, C, T, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''
        videos = videos.permute(0, 2, 1, 3, 4)
        bs, num_frames, num_channels, h, w = videos.shape
        videos = videos.flatten(0, 1)
        features = self.forward(videos)
        features = features.reshape(bs, -1, features.shape[2])

        return features
    
class ResNet50(nn.Module):
    def __init__(self):

        super().__init__()
        from transformers import ResNetModel
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.latent_dim = 8192


    def forward(self, images):
        '''
        images: [B, C, H, W], Image is normalized with imagenet norm
        '''
        input_dict = {'pixel_values': images}

        decoder_outputs = self.model(**input_dict, output_hidden_states=True)

        features = decoder_outputs.last_hidden_state
        
        features = features.reshape(features.shape[0], -1)
        
        features = nn.AdaptiveAvgPool1d(8192)(features.float())

        return features


class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import ViTMAEForPreTraining as automodel
        self.model = automodel.from_pretrained('facebook/vit-mae-huge', mask_ratio=0.0)#.to(device).eval()
        self.latent_dim = 8192

    def forward(self, images):
        '''
        images: [B, C, H, W], Image is normalized with imagenet norm
        '''
        input_dict = {'pixel_values': images}

        return_dict = self.model.config.use_return_dict

        outputs = self.model.vit(
            input_dict['pixel_values'],
            noise=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore

        decoder_outputs = self.model.decoder(latent, ids_restore, output_hidden_states=True)

        features = decoder_outputs.hidden_states[-4][:, 1:]

        features = features.reshape(features.shape[0], -1)
        
        features = nn.AdaptiveAvgPool1d((8192))(features.float())
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
    if encoder == "r3m":
        return R3M_pretrained
    elif encoder == "dino":
        return DINOV2
    elif encoder == "resnet":
        return ResNet50
    elif encoder == "mae":
        return MAE
    else:
        raise NotImplementedError(encoder)


def _get_dynamics(dynamics):
    if dynamics == "id":
        return ID
    elif dynamics == "lstm":
        return LSTM
    else:
        raise NotImplementedError(dynamics)

def pfR3M_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="r3m", dynamics="lstm", n_past=n_past, **kwargs
    )

def pfDINO_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="dino", dynamics="lstm", n_past=n_past, **kwargs
    )

def pfResNet_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="resnet", dynamics="lstm", n_past=n_past, **kwargs
    )

def pfMAE_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="mae", dynamics="lstm", n_past=n_past, **kwargs
    )

def pfR3M_ID(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="r3m", dynamics="id", n_past=n_past, **kwargs
    )
