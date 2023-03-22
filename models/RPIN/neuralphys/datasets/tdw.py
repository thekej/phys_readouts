import os
import random
import numpy as np
import tensorflow as tf
import torch

from physics.data.data_provider import SequenceNewDataProvider as DataProvider

from torch.utils.data import Dataset
from neuralphys.utils.bbox import nonlinear_transform
from neuralphys.utils.config import _C as C



def filter_rule(data, keys):
    assert all(k in keys for k in ['is_moving', 'is_acting']), keys
    return tf.logical_and(data['is_moving'], tf.logical_not(data['is_acting']))


DEFAULT_DATA_PARAMS = {
        'enqueue_batch_size': 200, #100, #256
        'map_pcall_num': 1,
        'buffer_size': 256 * 2,
        'batch_size': 1,
        'main_source_key': 'images',
        'sources': ['images', 'bounding_boxes', 'reference_ids'],
        'filter_rule': (filter_rule, ['is_moving', 'is_acting']),
        'seed': 0,
        'use_legacy_sequence_mode': 1,
        }


class TDWPhys(Dataset):
    def __init__(
            self,
            data_root,
            split,
            test = False,
            ):
        self.test = test
        self.preload_as_list = self.test
        self.data_path = data_root if isinstance(data_root, list) else [data_root]
        self.split = split
        self.data_path = [os.path.join(p, 'new_tfdata') if self.split == 'train' \
                else os.path.join(p, 'new_tfvaldata') for p in self.data_path]
        # 1. define the length of input and rollout sequences
        self.input_size = C.RPIN.INPUT_SIZE
        self.cons_size = C.RPIN.CONS_SIZE
        self.infer_start = self.input_size - self.cons_size
        self.pred_size = C.RPIN.PRED_SIZE_TRAIN if self.split == 'train' \
                else C.RPIN.PRED_SIZE_TEST
        # because we are predicting the 'next' frame, we need 1 offset
        self.buffer_size = self.input_size + self.pred_size + 1

        # 2. define output annotations
        self.pred_offset = C.RPIN.OFFSET_LOSS_WEIGHT > 0
        self.pred_position = C.RPIN.POSITION_LOSS_WEIGHT > 0

        # 3. define model configs
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH

        # build dataset
        data_params = DEFAULT_DATA_PARAMS
        data_params['data'] = self.data_path
        data_params['sources'] += C.INPUT.BINARY_LABELS
        data_params['shift_selector'] = slice(*C.INPUT.TRAIN_SLICE) if self.split == 'train' \
                else slice(*C.INPUT.VAL_SLICE)
        data_params['buffer_size'] = data_params['buffer_size'] if self.split == 'train' \
                else 64
        data_params['sequence_len'] = self.buffer_size
        data_params['test'] = self.test
        #TODO: Expose delta time parameter
        data_params['delta_time'] = 2
        self.data = self.build_data(data_params)


    def build_data(self, data_params):
        # Disable GPUs for Tensorflow
        tf.config.set_visible_devices([], 'GPU')
        data_provider = DataProvider(**data_params)
        dataset = data_provider.build_datasets(data_params['batch_size'])
        if self.preload_as_list:
            print("Preloading data...")
            data = list(dataset)
            '''
            data = iter(dataset)
            n = 0
            while(True):
                try:
                    next(data)
                    n += 1
                    print(n)
                except StopIteration:
                    print("Total", n)
                    raise StopIteration
            '''
            print("Preloaded %d sequences." % len(data))
        else:
            print("Using data iterator...")
            data = iter(dataset)
        return data


    def __len__(self):
        if not self.preload_as_list:
            if self.split == 'train':
                return C.INPUT.TRAIN_NUM
            else:
                return C.INPUT.VAL_NUM
            #raise ValueError("No len! Data provider is using iterator mode not list mode!")
        else:
            return len(self.data)


    def __iter__(self):
        if self.preload_as_list:
            raise ValueError("No iter! Data provider is using list mode not iterator mode!")
        data = next(self.data)
        return self.postprocess(data)


    def __getitem__(self, idx):
        if not self.preload_as_list:
            while(True):
                data = next(self.data)
                # Skip sequences which have no objects in at least one of the frames
                if not np.any(np.all(data['bounding_boxes'][..., -1].numpy() < 1,
                    axis=-1)):
                    break
            #raise ValueError("No getitem! Data provider is using iterator mode not list mode!")
        else:
            data = self.data[idx]
        return self.postprocess(data)


    def postprocess(self, data):
        # Get first batch element only
        images = data['images'][0]
        bboxes = data['bounding_boxes'][0]
        reference_ids = data['reference_ids'][0]
        human_prob = np.zeros((4,))

        raw_images = images.numpy().copy()
        raw_bboxes = bboxes.numpy().copy()
        reference_ids = reference_ids.numpy()
        binary_labels = np.concatenate(
                [data[label_key][0].numpy().copy() \
                        for label_key in C.INPUT.BINARY_LABELS], axis=-1)


        images = np.transpose(images.numpy().copy(), [0, 3, 1, 2]).astype(np.float32)
        assert images.shape[-1] == self.input_width, (images.shape[-1], self.input_width)
        assert images.shape[-2] == self.input_height, (images.shape[-2], self.input_height)

        rois = bboxes[..., :-1].numpy().copy()
        ignore_mask = np.all(bboxes[..., -1].numpy().copy() > 0, axis=0).astype(np.float32)

        # TODO Substract mean and divide by stddev after dividing by 255.0
        input_images = images[:self.input_size].copy()
        last_images = images[-1:].copy() if C.RPIN.VAE else input_images.copy()
        #input_images = images[:self.input_size]
        #last_images = images[-1:] if C.RPIN.VAE else input_images

        input_images -= np.reshape(C.INPUT.IMAGE_MEAN, [1, 3, 1, 1])
        input_images /= np.reshape(C.INPUT.IMAGE_STD, [1, 3, 1, 1])
        last_images -= np.reshape(C.INPUT.IMAGE_MEAN, [1, 3, 1, 1])
        last_images /= np.reshape(C.INPUT.IMAGE_STD, [1, 3, 1, 1])

        #input_images /= 255.0
        #last_images /= 255.0

        infer_length = self.pred_size + self.cons_size
        # there are three different boxes used in the model:
        # 1. rois used for region feature extraction
        # 2. src_rois and tar_rois, they are used for computing regression targets
        # normalize input box coordinate to input image scale
        # horizontal flip as data augmentation
        if random.random() > 0.5 and self.split == 'train' and C.RPIN.HORIZONTAL_FLIP:
            rois[..., [0, 2]] = self.input_width - rois[..., [2, 0]]
            input_images = np.ascontiguousarray(input_images[..., ::-1])

        if random.random() > 0.5 and self.split == 'train' and C.RPIN.VERTICAL_FLIP:
            rois[..., [1, 3]] = self.input_height - rois[..., [3, 1]]
            input_images = np.ascontiguousarray(input_images[..., ::-1, :])

        # Pad invalid objects
        num_objs = rois.shape[1]
        if num_objs < C.RPIN.NUM_OBJS:
            #rois = np.concatenate([rois, rois[:, :C.RPIN.NUM_OBJS - num_objs]], axis=1)
            rois = np.concatenate([rois,
                np.repeat(rois[:, 0:1], C.RPIN.NUM_OBJS - num_objs, axis = 1)], axis=1)
            ignore_mask = np.concatenate([ignore_mask,
                np.zeros((C.RPIN.NUM_OBJS - num_objs))])

        src_rois = rois[self.infer_start:][:infer_length].copy()
        tar_rois = rois[self.infer_start + 1:][:infer_length].copy()
        labels, dir_labels = nonlinear_transform(src_rois.reshape(-1, 4),
                tar_rois.reshape(-1, 4), False)

        labels[..., 0] /= self.input_width
        labels[..., 1] /= self.input_height
        labels = labels.reshape(infer_length, -1, 2)

        input_images = torch.from_numpy(input_images.astype(np.float32))
        last_images = torch.from_numpy(last_images.astype(np.float32))
        rois = torch.from_numpy(rois.astype(np.float32))
        ignore_mask = torch.from_numpy(ignore_mask.astype(np.float32))

        pos_labels = np.zeros(labels.shape, dtype=np.float32)
        pos_labels[..., 0] = np.sum(tar_rois[..., [0, 2]], axis=-1) / 2 / self.input_width
        pos_labels[..., 1] = np.sum(tar_rois[..., [1, 3]], axis=-1) / 2 / self.input_height
        labels = np.concatenate([labels, pos_labels], axis=-1)
        labels = torch.from_numpy(labels.astype(np.float32))

        if self.test:
            return input_images, rois, labels, last_images, ignore_mask, \
                    raw_images, raw_bboxes, binary_labels, reference_ids, human_prob
        else:
            return input_images, rois, labels, last_images, ignore_mask


    def get_valid_seq(self, image_list):
        return 1


    def get_video_info(self, valid_seq, idx):
        cur_video_info = np.zeros((valid_seq, 2), dtype=np.int32)
        cur_video_info[:, 0] = idx
        cur_video_info[:, 1] = np.arange(valid_seq)
        return cur_video_info