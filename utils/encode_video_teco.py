import argparse
import numpy as np
import os
import torch
import jax.random as random

from omegaconf import OmegaConf
from os import listdir
from os.path import isfile, join
from PIL import Image
from models.teco.teco.models import get_model, load_ckpt
from torchvision.transforms import functional as TF
import jax


def load_vqgan_model(checkpoint_path):
    kwargs = dict()
    
    model, state, config = load_ckpt(checkpoint_path, replicate=False,
                                     return_config=True, 
                                     **kwargs)
    return model, state, config

def encode_image(x, model, state):
    variables = {'params': state.params, **state.model_state}
    cond, out = model.apply(variables, x,
                            method=model.encode)
    devices = jax.devices()
    if 'gpu' in devices:
        print("JAX is using the GPU.")
    else:
        print("JAX is using the CPU.")

    #print(cond.shape, out.shape)
    return out

def main(args):
    model, state, config = load_vqgan_model(args.vqgan_checkpoint)
     
    def wrap_apply(fn):
        variables = {'params': state.params, **state.model_state}
        return lambda *args: model.apply(variables, *args, method=fn)

    video_encode = jax.jit(wrap_apply(model.encode))
    encoded_images = []
    images_path = [args.image_dir+f for f in listdir(args.image_dir) if isfile(join(args.image_dir, f))]
    for im_path in images_path:
        pil_image = Image.open(im_path).convert('RGB')
        pil_image = pil_image.resize((128, 128))
        pil_image = np.expand_dims(np.array(pil_image), axis=0)
        pil_image = jax.numpy.array(pil_image)
        p_encode = video_encode(pil_image)
        print(p_encode[1].shape)
        encoded_images += p_encode
    #np.save(args.save_path, encoded_images)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-co', '--vqgan-config', type=str, default=None,
                        help='Path for model config')
    parser.add_argument('-ckpt', '--vqgan-checkpoint', type=str, default=None,
                        help='Path for model checkpoint')
    parser.add_argument('-id', '--image-dir', type=str,
                        help='Path to save data on')
    parser.add_argument('-o', '--save-path', type=str,
                        default='encoded_data.',
                        help='Path to save data on')

    args = parser.parse_args()
    main(args)
