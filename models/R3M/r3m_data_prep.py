import argparse
import io
import glob
import h5py
import numpy as np
import os
import tqdm

from os import listdir
from os.path import isfile, join
from PIL import Image
from torchvision import transforms


R3M_VAL_TRANSFORMS = [transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),]

def read_video(video_file, data_transform, downsample_rate=6):
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
            if not i % downsample_rate == 0:
                continue
            img = f['frames'][frame]['images']['_img_cam0'][:]
            img = Image.open(io.BytesIO(img)) # (256, 256, 3)
            images.append(data_transform(img))

        ## apply frame rate normalization
        label = 1 if target_contacted_zone else 0
        stimulus = f['static']['stimulus_name'][()]
    return images, label, stimulus


def main(args):
    # set up new dataset
    videos = glob.glob(os.path.join(args.data_dir, "**/**/*.hdf5"))
    corrupt = glob.glob(os.path.join(args.data_dir, '**/**/temp.hdf5'))
    videos = list(set(videos) - set(corrupt))
    print('Number of videos: ',len(videos)) 
    vid_len = 25
    data_transform = transforms.Compose(R3M_VAL_TRANSFORMS)
    
    f = h5py.File(args.save_path, "w")
    dset1 = f.create_dataset("video", (len(videos), vid_len, 3, 224, 224), dtype='f')
    dset3 = f.create_dataset("label", (len(videos),), dtype='i')
    dset4 = f.create_dataset("stimulus", (len(videos),), dtype='i')
    
    
    stimulus_map = {}
    length = []
    idx = 0
    for i in tqdm.tqdm(range(len(videos))):
        images, labels, stimulus = read_video(videos[i], data_transform)
        if vid_len > len(images):
            images += [images[-1]] * (vid_len - len(images))
        elif vid_len < len(images):
            images = images[:vid_len]

        images = np.stack(images)

        if not stimulus in stimulus_map.keys():
            stimulus_map[str(stimulus)] = idx
            idx += 1
            
        dset1[i] = images
        dset3[i] = labels
        dset4[i] = stimulus_map[str(stimulus)]

    f.close()

    with open(args.map_dir, 'w') as f:
        import json
        json.dump(stimulus_map, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str,
                        help='Path to save data on')
    parser.add_argument('-o', '--save-path', type=str,
                        default='encoded_data.',
                        help='Path to save data on')
    parser.add_argument('-m', '--map-dir', type=str,
                        default='encoded_data.',
                        help='Path to save data on')

    args = parser.parse_args()
    main(args)
