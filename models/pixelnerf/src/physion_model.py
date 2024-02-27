import torch
import torch.nn as nn
from physion_feature_extraction.feature_extract_interface import PhysionFeatureExtractor
from physion_feature_extraction.utils import Stack, ToTorchFormatTensor, GroupNormalize, GroupScale
from physion_feature_extraction.utils import get_fourier_features
from pyhocon import ConfigFactory
from torchvision import transforms

from model import make_model


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


class PN(PhysionFeatureExtractor):
    def __init__(self, weights_path, model_name):

        super().__init__()

        conf = '../conf/exp/sn64.conf'

        conf = ConfigFactory.parse_file(conf)

        self.net = make_model(conf["model"])

        self.net.load_state_dict(torch.load(weights_path, map_location='cpu'))

    def transform(self, ):
        '''
        :return: Image Transform, Frame Gap, Minimum Number of Frames

        '''

        return Transform(rescale_size=64), 150, 17

    def extract_features(self, videos):
        '''
        videos: [B, T, C, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''

        bs, t, c, h, w = videos.shape

        videos = videos.flatten(0, 1)

        feature_map = self.net.encoder(videos)

        feature_map = feature_map.view(bs, t, feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])

        return feature_map
    
    def extract_features_ocd(self, videos):
        '''
        videos: [B, T, C, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''

        bs, t, c, h, w = videos.shape

        videos = videos.flatten(0, 1)

        feature_map = self.net.encoder(videos)

        #feature_map = feature_map.view(bs, t, feature_map.shape[2], feature_map.shape[2], feature_map.shape[1])
        
        #features_ocd = []
        #for i in range(t):
        #    fft_map = get_fourier_features(feature_map[0, i].unsqueeze(0))
        #    features_ocd += [fft_map]

        #feature_map = torch.stack(features_ocd, axis=1)
        features = feature_map.cpu().detach()
        features = features.reshape(t, -1)

        features = nn.AdaptiveAvgPool1d(8192)(features.float())
        features = features.view(bs, t, -1)
        return features


class PN_LSTM(PhysionFeatureExtractor):
    def __init__(self, weights_path, n_past=7, full_rollout=False):
        super().__init__()
        from r3m_model import pfPN_LSTM_physion, load_model
        self.model = pfPN_LSTM_physion(n_past=n_past, full_rollout=full_rollout)
        self.model = load_model(self.model, weights_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.n_past = n_past
        
    def transform(self):
        return Transform(rescale_size=64), 60, 25

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
    
class PN_LSTM_OCD(PN_LSTM):
    def __init__(self, weights_path):
        super().__init__(weights_path, full_rollout=True)

    def transform(self):
        return Transform(rescale_size=64), 60, 66

class PN_LSTM_SIM(PN_LSTM):
    def __init__(self, weights_path):
        super().__init__(weights_path, full_rollout=True)


class PixelNERF_shpnet(PN):
    def __init__(self, weights_path):
        super().__init__(weights_path, 'pixelnerf')
