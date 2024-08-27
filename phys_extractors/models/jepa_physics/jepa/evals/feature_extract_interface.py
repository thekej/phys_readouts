from abc import ABC, abstractmethod

from torchvision import transforms
from torch import nn

class ActivityRecogFeatureExtractor(ABC, nn.Module):

    def __int__(self):
        '''
        weights_path: path to the weights of the model
        '''

    @abstractmethod
    def get_embed_dim(self):

        return None

    @abstractmethod
    def get_num_heads(self):

        return None

    @abstractmethod
    def forward(self, videos):
        '''
        videos: [B, C, N, H, W], N is usually 100 for intphys and videos are normalized with imagenet norm
        returns: plausibility score for the video by using reconstruction losses on frame triplets
        '''


class Test(ActivityRecogFeatureExtractor):
    def __init__(self):
        super().__init__()

    def forward(self, videos):
        '''
        videos: [B, C, N, H, W], N is usually 100 for intphys and videos are normalized with imagenet norm
        returns: plausibility score for the video by using reconstruction losses on frame triplets
        '''

