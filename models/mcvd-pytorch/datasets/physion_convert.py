import argparse
import cv2
import glob
import imageio
import io
import numpy as np
import os
import sys
import h5py


from functools import partial
from multiprocessing import Pool
from PIL import Image
from tqdm import tqdm

from h5 import HDF5Maker


class UCF101_HDF5Maker(HDF5Maker):

    def create_video_groups(self):
        self.writer.create_group('len')
        self.writer.create_group('data')
        self.writer.create_group('target')
        self.writer.create_group('scenario')

    def add_video_data(self, data, scenario_class, dtype=None):
        data, target, scenario_class = data
        self.writer['len'].create_dataset(str(self.count), data=len(data))
        self.writer['target'].create_dataset(str(self.count), data=target, dtype='uint8')
        self.writer['scenario'].create_dataset(str(self.count), data=scenario_class, dtype='uint')
        self.writer.create_group(str(self.count))
        for i, frame in enumerate(data):
            self.writer[str(self.count)].create_dataset(str(i), data=frame, dtype=dtype, compression="lzf")


def read_video(video_file, original_fps=100, new_fps=25):
    
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
            pil_im_rsz = img.resize((64, 64), Image.LANCZOS)  
            images.append(pil_im_rsz)

        images = np.stack(images)
        #images = images[::(original_fps// new_fps)]
        ## apply frame rate normalization
        label = 1 if target_contacted_zone else 0
        stimulus = f['static']['stimulus_name'][()]
    return images, label, stimulus


def process_video(video_file):
    images = []
    label = None
    stimulus = ''
    try:
        images, label, stimulus = read_video(video_file)
    except StopIteration:
        pass
        # break
    except (KeyboardInterrupt, SystemExit):
        print("Ctrl+C!!")
        return "break"
    except:
        e = sys.exc_info()[0]
        print("ERROR:", video_file)
    return images, label, stimulus



def make_h5_from_ucf(data_dir, splits_dir, split_idx, image_size, out_dir='./h5_ds', vids_per_shard=100000, force_h5=False):

    # H5 maker
    h5_maker = UCF101_HDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)

    print(data_dir)
    vids_train = glob.glob(os.path.join(data_dir, "**/**/*.hdf5"))
    print("Train:", len(vids_train), "\nTest", len([]))

    h5_maker.writer.create_dataset('num_train', data=len(vids_train))
    h5_maker.writer.create_dataset('num_test', data=len([]))
    videos = vids_train
    
    stimulus_map = {}
    idx = 0
    for i in tqdm(range(len(videos))):
        images, labels, stimulus = process_video(videos[i])
        if isinstance(images, str) and images == "break":
            break
            
        #stimulus = str(stimulus).split('_')[1]
        if not stimulus in stimulus_map.keys():
            stimulus_map[str(stimulus)] = idx
            idx += 1
        #print(stimulus)
        h5_maker.add_data((images, labels, stimulus_map[str(stimulus)]), dtype='uint8')

    h5_maker.close()
    
    with open(args.map_dir, 'w') as f:
        import json
        json.dump(stimulus_map, f)
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help="Directory to save .hdf5 files")
    parser.add_argument('--map_dir', type=str, help="Directory to save json map for stimuli")
    parser.add_argument('--ucf_dir', type=str, help="Path to UCF-101 videos")
    parser.add_argument('--splits_dir', type=str, help="Path to ucfTrainTestlist")
    parser.add_argument('--split_idx', type=int, choices=[1, 2, 3], default=2, help="Which split to use")
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--vids_per_shard', type=int, default=100000)
    parser.add_argument('--force_h5', type=eval, default=False)

    args = parser.parse_args()

    make_h5_from_ucf(out_dir=args.out_dir, data_dir=args.ucf_dir, splits_dir=args.splits_dir, split_idx=args.split_idx,
                     image_size=args.image_size, vids_per_shard=args.vids_per_shard, force_h5=args.force_h5)
