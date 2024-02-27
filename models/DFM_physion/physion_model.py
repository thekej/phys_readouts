import torch
import torch.nn as nn
from einops import rearrange
from physion_feature_extraction.feature_extract_interface import PhysionFeatureExtractor
from physion_feature_extraction.utils import Stack, ToTorchFormatTensor, GroupNormalize, GroupScale
from physion_feature_extraction.utils import get_fourier_features
from torchvision import transforms
from torch.functional import F

class Transform(object):
    def __init__(self,
                 imagenet_normalize=True,
                 rescale_size=None):

        transform_list = []

        if rescale_size is not None:
            print("RESCALE", rescale_size)
            self.rescale = GroupScale(rescale_size)
            transform_list.append(self.rescale)

        transform_list.extend([Stack(roll=False),
                               ToTorchFormatTensor(div=True)])

        if imagenet_normalize:
            IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
            IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)
            normalize = GroupNormalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            transform_list.append(normalize)

        self.transform = transforms.Compose(transform_list)

    def __call__(self, images):

        images = (images, None)

        process_data, _ = self.transform(images)

        return process_data


class DFM(PhysionFeatureExtractor):
    def __init__(self, weights_path, model_name):

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

        ckpt = torch.load(weights_path, map_location='cpu')

        new_dict = {}
        for key in ckpt['model'].keys():
            if 'enc' in key:
                new_dict[key[6:]] = ckpt['model'][key]

        self.model.load_state_dict(new_dict, strict=False)

    def transform(self, ):
        '''
        :return: Image Transform, Frame Gap, Minimum Number of Frames

        '''

        return Transform(rescale_size=128), 150, 17

    def extract_features(self, videos):
        '''
        videos: [B, T, C, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''
        ctxt_rgb = videos

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

        feature_map = feature_map.view(b, num_context, feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])

        return feature_map

    def extract_features_ocd(self, videos):
        '''
        videos: [B, T, C, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''

        ctxt_rgb = videos

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
        features = F.interpolate(feature_map, size=(H_new, W_new), mode='bilinear', align_corners=False)
        # Now, we will use the interpolate function from the torch.nn.functional module
        features = features.cpu().detach()
        features = features.reshape(num_context, -1)
        features = nn.AdaptiveAvgPool1d(8192)(features.float())
        features = features.view(b, num_context, -1)
        return features

class DFM_re10k(DFM):
    def __init__(self, weights_path):
        super().__init__(weights_path, 're10k')


class DFM_LSTM(PhysionFeatureExtractor):
    def __init__(self, weights_path, n_past=7, full_rollout=False):
        super().__init__()
        from r3m_model import pfDFM_LSTM_physion, load_model
        self.model = pfDFM_LSTM_physion(n_past=n_past, full_rollout=full_rollout)
        self.model = load_model(self.model, weights_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.n_past = n_past
        
    def transform(self):
        return Transform(rescale_size=128), 60, 25

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
            output = self.model(videos)
        features = torch.cat([output["input_states"], output["observed_states"]], axis=1)
        return features
    
    def extract_features_sim(self, videos):
        sim_length = 25
        if sim_length > videos.shape[1]:
            added_frames = sim_lengh - videos.shape[1]
            videos = torch.cat([videos] + [videos[:, -1]]*added_frames, axis=1)
        
        with torch.no_grad():
            output = self.model(videos[:, :sim_length])
        features = output["states"]
        return features
    
class DFM_LSTM_OCD(DFM_LSTM):
    def __init__(self, weights_path):
        super().__init__(weights_path, full_rollout=True)

class DFM_LSTM_SIM(DFM_LSTM):
    def __init__(self, weights_path):
        super().__init__(weights_path, full_rollout=True)
