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

def process_video(video_file, video_length, return_length_only=False, original_fps=100, new_fps=25):
    try:
        with h5py.File(video_file, 'r') as f: # load ith hdf5 file from list
            frames = list(f['frames'])
            target_contacted_zone = False
            for frame in reversed(frames):
                lbl = f['frames'][frame]['labels']['target_contacting_zone'][()]
                if lbl: # as long as one frame touching, label is True
                    target_contacted_zone = True
                    break

            images = []
        
            for i, frame in enumerate(frames):
                if not i % (original_fps // new_fps) == 0:
                    continue
                img = f['frames'][frame]['images']['_img_cam0'][:]
                img = Image.open(io.BytesIO(img)) # (256, 256, 3)
                pil_image = img.resize((128, 128), Image.LANCZOS) 
                pil_image = np.expand_dims(np.array(pil_image), axis=0)
                pil_image = 2 * (pil_image / 255.0) - 1
                pil_image = jax.numpy.array(pil_image)
                images.append(pil_image)
        
            action = np.zeros(video_length)
            label = 1 if target_contacted_zone else 0
            stimulus = f['static']['stimulus_name'][()]
    except:
        images, label, action, stimulus, frames = [], [], [], [], []
    return images, label, action, stimulus, len(frames)

def main(args):
    # set up new dataset
    videos = glob.glob(os.path.join(args.data_dir, "**/*.hdf5"))
    corrupt = glob.glob(os.path.join(args.data_dir, '**/temp.hdf5'))
    videos = list(set(videos) - set(corrupt))
    vid_len = 60
    
    f = h5py.File(args.save_path, "w")
    dset1 = f.create_dataset("video", (len(videos), vid_len, 16, 16), dtype='f')
    dset2 = f.create_dataset("action", (len(videos), vid_len), dtype='i')
    dset3 = f.create_dataset("label", (len(videos),), dtype='i')
    dset4 = f.create_dataset("stimulus", (len(videos),), dtype='i')
    
    # load model
    model, state, config = load_vqgan_model(args.vqgan_checkpoint)
     
    def wrap_apply(fn):
        variables = {'params': state.params, **state.model_state}
        return lambda *args: model.apply(variables, *args, method=fn)

    video_encode = jax.jit(wrap_apply(model.encode))
    stimulus_map = {}
    length = []
    idx, video_idx = 0, 0
    for i in tqdm.tqdm(range(len(videos))):
        images, labels, actions, stimulus, l = process_video(videos[i], vid_len, True)
        if len(images) == 0:
            dset1[i] = dset1[i-1]
            dset2[i] = dset2[i-1]
            dset3[i] = dset3[i-1]
            dset4[i] = dset4[i-1]
            continue
        encoded_images = []
        for j, im in enumerate(images):
            if j < vid_len:
                p_encode = video_encode(im)
                encoded_images += [p_encode[1].squeeze(0)]
        if vid_len > len(encoded_images):
            encoded_images += [np.zeros_like(p_encode[1].squeeze(0))] * (vid_len - len(encoded_images))

        encoded_images = np.stack(encoded_images)

        if not stimulus in stimulus_map.keys():
            stimulus_map[str(stimulus)] = idx
            idx += 1
            
        dset1[i] = encoded_images
        dset2[i] = actions
        dset3[i] = labels
        dset4[i] = stimulus_map[str(stimulus)]
        #video_idx += 1

    f.close()

    with open(args.map_dir, 'w') as f:
        import json
        json.dump(stimulus_map, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--vqgan-checkpoint', type=str, default=None,
                        help='Path for model checkpoint')
    parser.add_argument('-id', '--data-dir', type=str,
                        help='Path to save data on')
    parser.add_argument('-o', '--save-path', type=str,
                        default='encoded_data.',
                        help='Path to save data on')
    parser.add_argument('-m', '--map-dir', type=str,
                        default='encoded_data.',
                        help='Path to save data on')

    args = parser.parse_args()
    main(args)
