import numpy as np
import torch
import torch.nn as nn

from physion_feature_extraction.feature_extract_interface import PhysionFeatureExtractor
from physion_feature_extraction.utils import get_fourier_features, DataAugmentationForVideoMAE

from collections import OrderedDict
from torchvision import transforms

import torch.nn.functional as F


class PhysionFeatureExtractor(nn.Module):

    def __int__(self, weights_path):
        '''
        weights_path: path to the weights of the model
        '''

    def extract_features(self, videos):
        '''
        videos: [B, T, C, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''

    def extract_features_for_seg(self, img):
        '''
        img: [B, C, H, W], Image is normalized with imagenet norm
        returns: [B, H, W, D] extracted features
        '''


class R3M(PhysionFeatureExtractor):
    def __init__(self, weights_path):
        from models.R3M.r3m_model import R3M_pretrained
        super().__init__()
        self.model = R3M_pretrained()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()

    def transform(self):
        return DataAugmentationForVideoMAE(False, 224), 40, 12
    
    def get_encoder_feats(self, x):
        # applies encoder to each image in x: (Bs, T, 3, H, W) or (Bs, 3, H, W)
        with torch.no_grad():  # TODO: best place to put this
            feats = []
            for _x in torch.split(x, 1, dim=1):
                _x = torch.squeeze(
                    _x, dim=1
                )  # _x is shape (Bs, 1, 3, H, W) => (Bs, 3, H, W) TODO: put this in _extract_feats?
                feats.append(self._extract_feats(_x))
        return torch.stack(feats, axis=1)

    def _extract_feats(self, x):
        feats = self.model(x)
        feats = torch.flatten(feats, start_dim=1)  # (Bs, -1)
        return feats

    def extract_features(self, videos):
        with torch.no_grad():
            output = self.get_encoder_feats(videos)
        return output

    def extract_features_ocd(self, videos):
        return self.extract_features(videos)
    

class R3M_LSTM(PhysionFeatureExtractor):
    def __init__(self, weights_path, n_past=7, full_rollout=False):
        from models.R3M.r3m_model import pfR3M_LSTM_physion, load_model 
        super().__init__()
        self.model = pfR3M_LSTM_physion(n_past=n_past, full_rollout=full_rollout)
        self.model = load_model(self.model, weights_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.n_past = n_past

    def transform(self):
        return DataAugmentationForVideoMAE(False, 224), 60, 25

    def extract_features(self, videos):
        with torch.no_grad():
            output = self.model(videos[:, -self.model.n_past:])
        features = output["input_states"]
        return features

    def extract_features_ocd(self, videos):
        if self.n_past > videos.shape[1]:
            added_frames = self.n_past - videos.shape[1]
            videos = torch.cat([videos] + [videos[:, -1]]*added_frames, axis=1)
        with torch.no_grad():
            output = self.model(videos, n_past=videos.shape[1])
        features = torch.cat([output["input_states"], output["observed_states"]], axis=1)
        return features

class R3M_LSTM_OCD(R3M_LSTM):
    def __init__(self, weights_path):
        super().__init__(weights_path, full_rollout=True)


class DINOV2_LSTM(PhysionFeatureExtractor):
    def __init__(self, weights_path, n_past=7, full_rollout=False):
        super().__init__()
        from models.R3M.r3m_model import pfDINO_LSTM_physion, load_model
        self.model = pfDINO_LSTM_physion(n_past=n_past, full_rollout=full_rollout)
        self.model = load_model(self.model, weights_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.n_past = n_past
        
    def transform(self):
        return DataAugmentationForVideoMAE(True, 224), 60, 25

    def extract_features(self, videos):
        with torch.no_grad():
            output = self.model(videos[:, -self.model.n_past:])
        features = output["input_states"]
        return features

    def extract_features_ocd(self, videos):
        if self.n_past > videos.shape[1]:
            added_frames = self.n_past - videos.shape[1]
            videos = torch.cat([videos] + [videos[:, -1]]*added_frames, axis=1)
        with torch.no_grad():
            output = self.model(videos, n_past=videos.shape[1])
        features = torch.cat([output["input_states"], output["observed_states"]], axis=1)
        return features
    
    
class DINOV2_LSTM_OCD(DINOV2_LSTM):
    def __init__(self, weights_path):
        super().__init__(weights_path, full_rollout=True)


from models.mcvd_pytorch.load_model_from_ckpt import load_model, get_readout_sampler, init_samples
from models.mcvd_pytorch.datasets import data_transform
from models.mcvd_pytorch.runners.ncsn_runner import conditioning_fn
    
class MCVD(PhysionFeatureExtractor):
    def __init__(self, weights_path, 
                 model_type='physion',
                 n_features=13):
        super().__init__()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # load model
        self.scorenet, self.config = load_model(weights_path, device)
        # get sampler
        self.sampler = get_readout_sampler(self.config)
        self.n_features = n_features
        self.model_type = model_type
        
    def transform(self):
        return DataAugmentationForVideoMAE(False, 64), 40, 37

    def extract_features(self, videos):
        #videos = torch.stack([self.transform_video_tensor(vid) for vid in videos])
        input_frames = data_transform(self.config, videos)
        if self.model_type == 'ucf':
            # repeat data
            real, cond, cond_mask = conditioning_fn(self.config, input_frames[:, 8:16, :, :, :],
                                                    num_frames_pred=self.config.data.num_frames,
                                                    prob_mask_cond=getattr(self.config.data, 'prob_mask_cond', 0.0),
                                                    prob_mask_future=getattr(self.config.data, 'prob_mask_future', 0.0))
        else:
            real, cond, cond_mask = conditioning_fn(self.config, input_frames[:, -(self.config.data.num_frames_cond+self.config.data.num_frames):, :, :, :],
                                                    num_frames_pred=self.config.data.num_frames,
                                                    prob_mask_cond=getattr(self.config.data, 'prob_mask_cond', 0.0),
                                                    prob_mask_future=getattr(self.config.data, 'prob_mask_future', 0.0))
        init = init_samples(len(real), self.config)
        with torch.no_grad():
            pred, gamma, beta, mid = self.sampler(init, self.scorenet, cond=cond,
                                         cond_mask=cond_mask,
                                         subsample=100, verbose=True)
        return mid

    def extract_features_ocd(self, videos):
        #videos = torch.stack([self.transform_video_tensor(vid) for vid in videos])
        input_frames = data_transform(self.config, videos)
        if self.config.data.num_frames_cond+self.config.data.num_frames > videos.shape[1]:
            added_frames = self.config.data.num_frames_cond+self.config.data.num_frames - videos.shape[1]
            input_frames = torch.cat([input_frames] + [input_frames[:, -1].unsqueeze(1)]*added_frames, axis=1)
        output = []
        for j in range(0, videos.shape[1], self.config.data.num_frames_cond+self.config.data.num_frames):
            if j + self.config.data.num_frames_cond+self.config.data.num_frames > videos.shape[1]:
                break
            real, cond, cond_mask = conditioning_fn(self.config, 
                                                    input_frames[:, 
                                                    j:j+self.config.data.num_frames_cond+self.config.data.num_frames, 
                                                    :, :, :], 
                                        num_frames_pred=self.config.data.num_frames,
                                        prob_mask_cond=getattr(self.config.data, 'prob_mask_cond', 0.0),
                                        prob_mask_future=getattr(self.config.data, 'prob_mask_future', 0.0))

            init = init_samples(len(real), self.config)
            with torch.no_grad():
                pred, gamma, beta, mid = self.sampler(init, self.scorenet, cond=cond,
                                         cond_mask=cond_mask,
                                         subsample=100, verbose=True)
            output +=  [mid]
        return torch.stack(output, axis=1)
    
class MCVD_PHYSION(MCVD):
    def __init__(self, weights_path):
        super().__init__(weights_path, model_type='physion')
        
class MCVD_EGO4D(MCVD):
    def __init__(self, weights_path):
        super().__init__(weights_path, model_type='ego4d')
        
class MCVD_UCF(MCVD):
    def __init__(self, weights_path):
        super().__init__(weights_path, model_type='ucf')


class FITVID(PhysionFeatureExtractor):
    # input is (Bs, T, 3, H, W)
    def __init__(self, weights_path, n_past=7):
        super().__init__()
        from models.FitVid import fitvid
        
        # IMPORTANT: n_past decides if OCP or OCD
        model = fitvid.FitVid(n_past=n_past, train=False)
        params = torch.load(weights_path, map_location="cpu")

        new_sd = OrderedDict()
        for k, v in params.items():
            name = k[7:] if k.startswith("module.") else k
            new_sd[name] = v
        model.load_state_dict(new_sd)
        print(f"Loaded parameters from {weights_path}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device)
        self.model.eval()
        
    def transform(self):
        return DataAugmentationForVideoMAE(False, 64), 60, 25

    def extract_features(self, videos):
        with torch.no_grad():
            output = self.model(videos, n_past=7)
        features = output['h_preds']
        return features

    def extract_features_ocd(self, videos): 
        with torch.no_grad():
            output = self.model(videos, n_past=videos.shape[1])
        features = output['h_preds']
        return features


class ResNet50(PhysionFeatureExtractor):
    def __init__(self, weights_path):

        super().__init__()
        from transformers import ResNetModel

        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")

    def transform(self,):
        return DataAugmentationForVideoMAE(
            imagenet_normalize=True,
            rescale_size=224,
        ), 100, 24

    def fwd(self, images):
        '''
        images: [B, C, H, W], Image is normalized with imagenet norm
        '''
        input_dict = {'pixel_values': images}

        decoder_outputs = self.model(**input_dict, output_hidden_states=True)

        features = decoder_outputs.last_hidden_state

        return features


    def extract_features(self, videos):
        '''
        videos: [B, C, T, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, C, H, W] extracted features
        '''
        bs, num_frames, num_channels, h, w = videos.shape
        videos = videos.flatten(0, 1)
        features = self.fwd(videos)
        features = features.reshape(bs, num_frames, features.shape[1], features.shape[2], features.shape[3])
        return features

    def extract_features_ocd(self, videos):
        '''
        videos: [1, T, C, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [1, T, H, W, D] extracted features
        '''

        videos = videos.flatten(0, 1)
        features = self.fwd(videos)
        features = features.unsqueeze(0)
        return features

    def extract_features_for_seg(self, img):
        '''
        img: [B, C, H, W], Image is normalized with imagenet norm
        returns: [B, H, W, D] extracted features
        '''
        feat = self.fwd(img)

        feat = feat.permute(0, 2, 3, 1)

        return feat
