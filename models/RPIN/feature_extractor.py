from skimage.draw import line_aa
from torch.utils.data import Dataset, DataLoader

from neuralphys.models.rpin import Net
from physion_loader import RPINDataset

import argparse
import glob
import io
import imageio
import logging
import numpy as np
import os.path as osp
import os
import tqdm 
import h5py
import torch
import torch.nn.functional as F


def init_rois(boxes, shape):
    batch, time_step, _, height, width = shape
    max_objs = boxes.shape[2]
    # coor features, normalized to [0, 1]
    num_im = batch * time_step
    # noinspection PyArgumentList
    co_f = np.zeros(boxes.shape[:-1] + (2,))
    co_f[..., 0] = torch.mean(boxes[..., [0, 2]], dim=-1).numpy().copy() / width
    co_f[..., 1] = torch.mean(boxes[..., [1, 3]], dim=-1).numpy().copy() / height
    coor_features = torch.from_numpy(co_f.astype(np.float32))
    rois = boxes[:, :time_step]
    batch_rois = np.zeros((num_im, max_objs))
    batch_rois[np.arange(num_im), :] = np.arange(num_im).reshape(num_im, 1)
    # noinspection PyArgumentList
    batch_rois = torch.FloatTensor(batch_rois.reshape((batch, time_step, -1, 1)))
    rois = torch.cat([batch_rois, rois], dim=-1)
    return rois, coor_features

def load_model(model, model_file):
    assert os.path.isfile(model_file), f'Cannot find model file: {model_file}'
    model = model.load_state_dict(torch.load(model_file))
    logging.info(f'Loaded existing ckpt from {model_file}')
    return model

def save_vis(frames, rois, labels, bbox, ignore_idx, stimulus_name, 
             output_dir, prefix=0, artifact_path='videos', n_vis=1):
    # print(frames.shape, rois.shape, labels.shape, bbox.shape)
    BS, T, _, H, W = frames.shape
    fps = 30 / 9
    for i in range(min(n_vis, BS)):
        curr_stim = stimulus_name[i]
        if type(curr_stim) == bytes:
            curr_stim = curr_stim.decode('utf-8')
        # labels = build_labels(rois[i].numpy()).astype(np.uint8)
        arr = []
        images = torch.permute(255*frames[i], (0,2,3,1)).numpy().astype(np.uint8)
        n_objs = int(ignore_idx[i].sum().item()) # ignore_idx: (BS, K)
        for t in range(T):
            image = images[t]
            for k in range(n_objs):
                off = T-labels.shape[1]
                if t >= off:
                    x,y = labels[i,t-off,k,-2:].numpy().astype(np.uint8)
                    image[y-1:y+1,x-1:x+1] = np.ones((1,3)) * 255
                    x, y = bbox[i, t-off,k,-2:].numpy().astype(np.uint8)
                    image[y-2:y+2,x-2:x+2] = np.array([[255,0,0]])
                image = add_bbox(image, rois[i,t,k])
            arr.append(image)
        arr = np.stack(arr)

        fn = os.path.join(output_dir, f'{prefix:06}_{i:02}_{curr_stim}.mp4')
        imageio.mimwrite(fn, arr, fps=fps)
        mlflow.log_artifact(fn, artifact_path=artifact_path)

def add_bbox(image, roi, color=np.array([255,0,0])):
    x1, y1, x2, y2 = roi.numpy().astype(np.uint8)
    if x1==x2 and y1==y2:
        return image
    image = add_line(image, y1, x1, y1, x2)
    image = add_line(image, y2, x1, y2, x2)
    image = add_line(image, y1, x1, y2, x1)
    image = add_line(image, y1, x2, y2, x2)
    return image

def add_line(image, y1, x1, y2, x2):
    # rr, cc, val = weighted_line(y1, x1, y2, x2, 5)
    rr, cc, val = line_aa(y1, x1, y2, x2)
    image[rr, cc] = val.reshape(-1,1).astype(np.uint8) * 255
    return image

def extract_feat_step(model, data, device, num_rollouts,
                      output_dir, save):
    model.eval()
    with torch.no_grad():
        labels = data['binary_labels'].cpu().numpy()
        stimulus_name = np.array(data['stimulus_name'], dtype=object)
        images, boxes, data_last, ignore_idx = [data[k] for k in ['images', 'rois', 'data_last', 'ignore_mask']]
        images = images.to(device)
        rois, coor_features = init_rois(boxes, images.shape)
        rois = rois.to(device)
        coor_features = coor_features.to(device)
        ignore_idx = ignore_idx.to(device)
        outputs = model(images, rois, coor_features, 
                        num_rollouts=num_rollouts,
                        data_pred=data_last, phase='test', 
                        ignore_idx=ignore_idx)
    input_states = torch.flatten(outputs['input_states'], 2).cpu().numpy()
    observed_states = torch.flatten(outputs['encoded_states'], 2).cpu().numpy()
    simulated_states = torch.flatten(outputs['rollout_states'], 2).cpu().numpy()

    if save:
        save_vis(data['images'], data['rois'], data['labels'], 
             outputs['bbox'].cpu(), ignore_idx, stimulus_name, 
             output_dir)

    output = {
        'input_states': input_states,
        'observed_states': observed_states,
        'simulated_states': simulated_states,
        'labels': labels,
        'stimulus_name': stimulus_name,
        }
    # print([(k,v.shape) for k,v in output.items()])
    return output

def main(args):
    print('load model')
    model = Net()
    model = load_model(model, args.model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    print('load data')
    dataset = RPINDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    data_size = 0
    for b in loader:
        v_in, label_in = b
        data_size += v_in.shape[0]
        
    n_features = 10
    # set up new dataset
    f = h5py.File(args.save_file, "w")
    dset1 = f.create_dataset("label", (data_size,), dtype='f')
    dset2 = f.create_dataset("observed", (data_size, 7, 2048), dtype='f')
    dset3 = f.create_dataset("observed_full_outcome", (data_size, 18, 2048), dtype='f')
    dset4 = f.create_dataset("simulation", (data_size, 18, 2048), dtype='f')

    print('start extraction')
    for i, data in enumerate(tqdm.tqdm(loader)):        
        output = extract_feat_step(model, data, device, args.num_rollouts,
                                   args.output_dir, args.save)
        
        if (i+1)*args.batch_size < data_size:
            dset1[i*args.batch_size:(i+1)*args.batch_size] = output["labels"]
        else:
            dset1[i*args.batch_size:] = output["labels"]
        # input is (Bs, T, 3, H, W)
        output = model(v_in)
        if (i+1)*args.batch_size < data_size:#(8, 4, 50, 16, 16) (8, 4, 50, 8, 8, 256) (8, 4, 5, 8, 8, 256)
            dset2[i*args.batch_size:(i+1)*args.batch_size] = output["input_states"].detach().cpu().numpy()
            dset3[i*args.batch_size:(i+1)*args.batch_size] = output["observed_states"].detach().cpu().numpy()
            dset4[i*args.batch_size:(i+1)*args.batch_size] = output["simulated_states"].detach().cpu().numpy()
        else:
            dset2[i*args.batch_size:] = output["input_states"].detach().cpu().numpy()
            dset3[i*args.batch_size:] = output["observed_states"].detach().cpu().numpy()
            dset4[i*args.batch_size:] = output["simulated_states"].detach().cpu().numpy()
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_rollouts', type=int, default=18)
    parser.add_argument('--save', action="store_false")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    main(args)