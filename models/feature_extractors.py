import numpy as np
import torch.nn as nn
import get_fourier_features

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



class DINOV2(PhysionFeatureExtractor):
    def __init__(self, weights_path, model_name):
        super().__init__()
        from transformers import AutoModel as automodel
        self.model = automodel.from_pretrained('facebook/' + model_name)#.to(device).eval()

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
        returns: [B, T, D] extracted features
        '''
        videos = videos.permute(0, 2, 1, 3, 4)[:, :4]
        bs, num_frames, num_channels, h, w = videos.shape
        videos = videos.flatten(0, 1)
        features = self.fwd(videos)

        patch_size = int(np.sqrt(features.shape[1]-1))
        features = features[:, 1:].view(features.shape[0], patch_size, patch_size, -1)
        # Add Fourier features
        features = get_fourier_features(features)

        return features

    def extract_features_for_seg(self, img):
        '''
        img: [B, C, H, W], Image is normalized with imagenet norm
        returns: [B, H, W, D] extracted features
        '''
        feat = self.fwd(img)

        ps = int(np.sqrt(feat.shape[1]-1))
        feat = feat[:, 1:].view(feat.shape[0], ps, ps, -1)

        return feat

class DINOV2Base(DINOV2):
    def __init__(self, weights_path):
        super().__init__(weights_path, 'dinov2-base')
        
        
class R3M_LSTM(PhysionFeatureExtractor):
    def __init__(self, weights_path):
        super().__init__()
        from models.R3M.r3m_model import pfR3M_LSTM_physion, load_model
        self.model = pfR3M_LSTM_physion()
        self.model = load_model(self.model, weights_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.data_transform = transforms.Compose(
                                    [transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),]
                                    )

    def extract_features(self, videos):
        videos = torch.stack([self.data_transform(img) for img in videos])
        output = model(videos)
        features = output["input_states"].detach().cpu().numpy()
        features = features.view(features.shape[0], -1, 32)
        patch_size = int(np.sqrt(features.shape[1]))
        features = features.view(features.shape[0], patch_size, patch_size, -1)
        return get_fourier_features(features)

    def extract_features_ocd(self, img):
        videos = torch.stack([self.data_transform(img) for img in videos])
        output = model(videos)
        features = output["observed_states"].detach().cpu().numpy()
        features = features.reshape(features.shape[0], -1, 32)
        patch_size = int(np.sqrt(features.shape[1]))
        features = features.reshape(features.shape[0], patch_size, patch_size, -1)
        return get_fourier_features(features)

    
class DINOV2_LSTM(PhysionFeatureExtractor):
    def __init__(self, weights_path):
        super().__init__()
        from models.R3M.r3m_model import pfDINO_LSTM_physion, load_model
        self.model = pfDINO_LSTM_physion()
        self.model = load_model(self.model, weights_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.data_transform = transforms.Compose(
                                    [transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),]
                                    )

    def extract_features(self, videos):
        videos = torch.stack([self.data_transform(img) for img in videos])
        output = model(videos)
        features = output["input_states"].detach().cpu().numpy()
        features = features.reshape(features.shape[0], -1, 32)
        patch_size = int(np.sqrt(features.shape[1]))
        features = features.reshape(features.shape[0], patch_size, patch_size, -1)
        return get_fourier_features(features)

    def extract_features_ocd(self, videos):
        videos = torch.stack([self.data_transform(img) for img in videos])
        output = model(videos)
        features = output["observed_states"].detach().cpu().numpy()
        features = features.reshape(features.shape[0], -1, 32)
        patch_size = int(np.sqrt(features.shape[1]))
        features = features.reshape(features.shape[0], patch_size, patch_size, -1)
        return get_fourier_features(features)
    
    
class MCVD(PhysionFeatureExtractor):
    def __init__(self, weights_path, 
                 model_type='physion',
                 n_features=1, 
                 n_context=13):
        super().__init__()
        from models.mcvd_pytoch.load_model_from_ckpt import load_model, get_readout_sampler, init_samples
        from models.mcvd_pytoch.datasets import data_transform
        from models.mcvd_pytoch.runners.ncsn_runner import conditioning_fn
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # load model
        self.scorenet, self.config = load_model(weights_path, device)
        # get sampler
        self.sampler = get_readout_sampler(self.config)
        self.n_features = n_features
        self.n_context = n_context
        self.data_transform = transforms.Compose(
                                [transforms.Resize(64), 
                                 transforms.ToTensor(),])


    def extract_features(self, videos):
        videos = torch.stack([self.data_transform(img) for img in videos])
        input_frames = data_transform(self.config, videos)
        if self.model_type == 'ucf':
            real, cond, cond_mask = conditioning_fn(self.config, input_frames[:, 8:16, :, :, :],
                                                    num_frames_pred=self.config.data.num_frames,
                                                    prob_mask_cond=getattr(self.config.data, 'prob_mask_cond', 0.0),
                                                    prob_mask_future=getattr(self.config.data, 'prob_mask_future', 0.0))
        else:
            real, cond, cond_mask = conditioning_fn(self.config, input_frames[:, :self.n_context, :, :, :],
                                                    num_frames_pred=self.config.data.num_frames,
                                                    prob_mask_cond=getattr(self.config.data, 'prob_mask_cond', 0.0),
                                                    prob_mask_future=getattr(self.config.data, 'prob_mask_future', 0.0))
        init = init_samples(len(real), self.config)
        pred, gamma, beta, mid = self.sampler(init, self.scorenet, cond=cond,
                                         cond_mask=cond_mask,
                                         subsample=100, verbose=True)
        #TODO: Add FFT aggregation
        feature = mid.numpy()
        features = features.reshape(mid.shape[0], -1, 32)
        patch_size = int(np.sqrt(features.shape[1]))
        features = features.reshape(features.shape[0], patch_size, patch_size, -1)
        return get_fourier_features(features)

    def extract_features_ocd(self, videos):
        videos = torch.stack([self.data_transform(img) for img in videos])
        input_frames = data_transform(self.config, videos)
        output = []
        for j in range(self.n_features):
            real, cond, cond_mask = conditioning_fn(config, input_frames[:, 4*j:4*j+8, :, :, :], 
                                                    num_frames_pred=config.data.num_frames,
                                        prob_mask_cond=getattr(config.data, 'prob_mask_cond', 0.0),
                                        prob_mask_future=getattr(config.data, 'prob_mask_future', 0.0))

            init = init_samples(len(real), config)
            pred, gamma, beta, mid = sampler(init, scorenet, 
                                             cond=cond, 
                                             cond_mask=cond_mask, 
                                             subsample=args.sub, 
                                             verbose=True)
            #TODO: Add FFT aggregation
            
            output +=  [mid.numpy()]
        features = np.stack(features, dim=1)
        features = features.reshape(mid.shape[0], -1, 32)
        patch_size = int(np.sqrt(features.shape[1]))
        features = features.reshape(features.shape[0], patch_size, patch_size, -1)
        return get_fourier_features(features)


class FITVID(PhysionFeatureExtractor):
    def __init__(self, weights_path, n_past=7):
        super().__init__()
        from models.FITVID import FitVid
        # IMPORTANT: n_past decides if OCP or OCD
        model = FitVid(n_past=n_past, train=False)
        params = torch.load(self.model_path, map_location="cpu")

        new_sd = OrderedDict()
        for k, v in params.items():
            name = k[7:] if k.startswith("module.") else k
            new_sd[name] = v
        model.load_state_dict(new_sd)
        print(f"Loaded parameters from {self.model_path}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device)
        self.data_transform = transforms.Compose(
                        [transforms.Resize(64), transforms.ToTensor(),])


    def extract_features(self, videos):
        videos = torch.stack([self.data_transform(img) for img in videos])
        output = self.model(videos)
        features = output['h_preds']
        features = features.reshape(mid.shape[0], -1, 32)
        patch_size = int(np.sqrt(features.shape[1]))
        features = features.reshape(features.shape[0], patch_size, patch_size, -1)
        return get_fourier_features(features)

    def extract_features_ocd(self, videos):
        return self.extract_features(videos)