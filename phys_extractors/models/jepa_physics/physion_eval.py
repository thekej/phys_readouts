from physion_feature_extraction.feature_extract_interface import PhysionFeatureExtractor
from physion_feature_extraction.utils import DataAugmentationForVideoMAE

import torch

class VJEPA(PhysionFeatureExtractor):
    def __init__(self, weights_path, model_name):
        super().__init__()
        import jepa.src.models.vision_transformer as vit
        import torch

        # download the model and put it in the folder.
        state_dict = torch.load(weights_path)

        # following the config for the model
        crop_size = 224
        patch_size = 16
        num_frames = 16
        tubelet_size = 2

        uniform_power = True
        use_sdpa = True
        use_SiLU = False
        tight_SiLU = False

        self.encoder = vit.__dict__[model_name](
            img_size=crop_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            uniform_power=uniform_power,
            use_sdpa=use_sdpa,
            use_SiLU=use_SiLU,
            tight_SiLU=tight_SiLU,
        )

        self.encoder.load_state_dict({k.replace('module.backbone.', ''): v for k, v in state_dict['encoder'].items()})
        self.encoder.eval()

    def transform(self):
        return DataAugmentationForVideoMAE(
            imagenet_normalize=True,
            rescale_size=224,
        ), 150, 4

    def extract_features(self, videos):
        '''
        videos: [B, T, C, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''

        videos = videos.transpose(1, 2)

        out = self.encoder(videos)

        return out

    def extract_features_for_seg(self, img):
        '''
        img: [B, C, H, W], Image is normalized with imagenet norm
        returns: [B, H, W, D] extracted features
        '''
        img = [img] * self.model.num_frames
        img = torch.stack(img, dim=2)  # [B, C, T, H, W]


        return img

class VJEPA_huge(VJEPA):
    def __init__(self, weights_path):
        super().__init__('./vith16.pth.tar', 'vit_huge')


class VJEPA_large(VJEPA):
    def __init__(self, weights_path):
        super().__init__('./vitl16.pth.tar', 'vit_large')