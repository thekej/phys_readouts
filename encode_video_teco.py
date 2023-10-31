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

import numpy as np

def get_contact(f, frame):
    cam_matrix = f['frames'][frame]['camera_matrices']['camera_matrix_cam0'][:]
    cam_matrix = cam_matrix.reshape(4, 4)
    proj_matrix = f['frames'][frame]['camera_matrices']['projection_matrix_cam0'][:]
    proj_matrix = proj_matrix.reshape(4, 4)
    contacts = proj_world_to_pixel(np.array(f['frames'][frame]['collisions']['contacts_ot'][()]), cam_matrix, proj_matrix)
    pts = np.array([[contacts[0][0], contacts[0][1]]])
    return pts

def pad_ones(pts):
    '''
    pts: [N , K]
    returns: [N, K+1]
    '''
    pw = np.concatenate([pts, np.ones([pts.shape[0], 1])], 1)
    return pw

def proj_world_to_pixel(pw, cam_matrix, proj_matrix):
    '''
    pw: [N, 3] world coords
    cam_matrix: [4, 4] cam matrix
    proj_matrix: [4, 4] proj matrix
    returns: [N, 2] pixel coords
    '''
    matrix = np.matmul(proj_matrix, cam_matrix)
    pw = pad_ones(pw).T
    proj_pts = np.matmul(matrix, pw).T
    proj_pts = proj_pts/proj_pts[:, 3:4]
    proj_pts = proj_pts.clip(-1, 1)
    proj_pts = (proj_pts + 1)/2
    proj_pts = proj_pts[:, :2]
    proj_pts[:, 1] = 1 - proj_pts[:, 1]
    proj_pts = proj_pts[:, [1, 0]]
    proj_pts = (proj_pts*128).astype(int)
    return proj_pts

def get_label(f):
#     try:
    with h5py.File(f) as h5file:
        stimulus = f['static']['stimulus_name'][()]
        for key in h5file['frames'].keys():
            lbl = np.array(h5file['frames'][key]['labels']['target_contacting_zone']).item()
            if lbl:
                contact = get_contact(h5file, key)
                if 'collide' in stimulus:
                    return int(key) + 7, True, contact
                else:
                    return int(key), True, contact

        ind = len(h5file['frames'].keys()) // 2

        return ind, False, np.array([[-1, -1]])


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
    with h5py.File(video_file, 'r') as f: # load ith hdf5 file from list
        frames = list(f['frames'])
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
        stimulus = f['static']['stimulus_name'][()]
        if 'collide' in stimulus:
            images = 7 * [images[0]] + images
    return images, action

def main(args):
    # set up new dataset
    videos = glob.glob(os.path.join(args.data_dir, "**/*.hdf5"))
    corrupt = glob.glob(os.path.join(args.data_dir, '**/temp.hdf5'))
    videos = list(set(videos) - set(corrupt))
    min_vid_len = 12
    
    #f = h5py.File(args.save_path, "w")
    dset1 = []# f.create_dataset("video", (len(videos), vid_len, 16, 16), dtype='f')
    dset2 = []#f.create_dataset("action", (len(videos), vid_len), dtype='i')
    dset3 = []#f.create_dataset("label", (len(videos),), dtype='i')
    dset4 = []#f.create_dataset("stimulus", (len(videos),), dtype='i')
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
    for i in tqdm.tqdm(range(len(videos))):
        collision_ind, label, contact = get_label(videos[i])
        images, actions = process_video(videos[i], vid_len, True)
        encoded_images = []
        for j, im in enumerate(images):
            #if j < vid_len:
            p_encode = video_encode(im)
            encoded_images += [p_encode[1].squeeze(0)]
        if min_vid_len > len(encoded_images):
            encoded_images += [p_encode[1].squeeze(0)] * (vid_len - len(encoded_images))

        encoded_images = np.stack(encoded_images)
        
        dset1 += [encoded_images]
        dset2 += [actions]
        dset3 += [label]
        dset4 += [contact]
        dset5 += [colision_ind]
        stimulus_list += stimulus_map[str(stimulus)]
     
    dt = h5py.special_dtype(vlen=str)

    with h5py.File(args.save_path ,'w') as hf:
        hf.create_dataset("video", data=np.concatenate(dset1))
        hf.create_dataset("action", data=np.concatenate(dset2))
        hf.create_dataset("label", data=np.array(dset3))
        hf.create_dataset("contacts", data=np.concatenate(dset4))  
        hf.create_dataset("collision_ind", data=np.array(dset5))  
        hf.create_dataset("stimulus", data=stimulus_list, dtype=dt)


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
