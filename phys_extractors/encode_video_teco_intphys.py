import argparse
import io
import jax.random as random
import glob
import h5py
import numpy as np
import os
import tqdm

from omegaconf import OmegaConf
from os import listdir
from os.path import isfile, join
from PIL import Image
from models.teco.teco.models import get_model, load_ckpt
import jax
from collections import defaultdict

import numpy as np


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
    print(devices)
    if 'gpu' in devices:
        print("JAX is using the GPU.")
    else:
        print("JAX is using the CPU.")

    return out

def process_video(img_paths):
    video = []
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        pil_image = img.resize((128, 128), Image.LANCZOS) 
        pil_image = np.expand_dims(np.array(pil_image), axis=0)
        pil_image = 2 * (pil_image / 255.0) - 1
        pil_image = jax.numpy.array(pil_image)
        video.append(pil_image)
    action = np.zeros(len(video))
    return video, action

def correct_path(st):

    splits = st.split('/')

    splits[1] = splits[1].split('_')[0]

    return '/'.join(splits).strip()

def get_video_paths(data_dir, video_dirs):
    video_paths = defaultdict(list)
    for video_dir in video_dirs:
        img_paths = sorted(
            glob.glob(os.path.join(data_dir, correct_path(video_dir), 'scene', 'scene_*.png')))
        #             print(len(sorted(glob.glob(os.path.join(self.data_dir, self.correct_path(video_dir), 'scene', 'scene_*.png')))))
        video_paths[video_dir.strip()] = img_paths
    return video_paths

def main(args):
    # set up new dataset
    data_dir = '/ccn2/u/rmvenkat/data/intphys/dev/'

    video_dirs = open('/ccn2/u/rmvenkat/data/intphys/dev/task.txt').readlines()

    videos = get_video_paths(data_dir, video_dirs)
    
    dset1 = []# f.create_dataset("video", (len(videos), vid_len, 16, 16), dtype='f')
    dset2 = []#f.create_dataset("action", (len(videos), vid_len), dtype='i')
    stimulus_list = []
    
    # load model
    model, state, config = load_vqgan_model(args.vqgan_checkpoint)
     
    def wrap_apply(fn):
        variables = {'params': state.params, **state.model_state}
        return lambda *args: model.apply(variables, *args, method=fn)

    video_encode = jax.jit(wrap_apply(model.encode))
    stimulus_map = {}
    length = []
    idx, video_idx = 0, 0
    for i in videos:
        print(i)
        images, actions = process_video(videos[i])
        encoded_images = []
        for j, im in enumerate(images):
            p_encode = video_encode(im)
            encoded_images += [p_encode[1].squeeze(0)]

        encoded_images = np.stack(encoded_images)
        dset1 += [encoded_images]
        dset2 += [actions]
        stimulus_list += [i.strip()]
        
    dt = h5py.special_dtype(vlen=str)

    with h5py.File(args.save_path ,'w') as hf:
        hf.create_dataset("video", data=np.stack(dset1))
        hf.create_dataset("action", data=np.stack(dset2))
        hf.create_dataset("filenames", data=stimulus_list, dtype=dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--vqgan-checkpoint', type=str, default=None,
                        help='Path for model checkpoint')
    parser.add_argument('-o', '--save-path', type=str,
                        default='encoded_data.',
                        help='Path to save data on')

    args = parser.parse_args()
    main(args)
