import os
import io
import glob
import h5py
import json
from PIL import Image
import numpy as np
import logging
import torch
from  torch.utils.data import Dataset
from torchvision import transforms

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.ERROR)  

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
        stimulus = h5file['static']['stimulus_name'][()]
        counter = 0
        for i, key in enumerate(h5file['frames'].keys()):
            if not i % 4 == 0:
                continue
            counter += 1
            lbl = np.array(h5file['frames'][key]['labels']['target_contacting_zone']).item()
            if lbl:
                contact = get_contact(h5file, key)
                
                if 'collide' in str(stimulus):
                    return counter + 12, True, contact
                else:
                    return counter, True, contact

        ind = len(h5file['frames'].keys()) // 8

        return ind, False, np.array([[-1, -1]])

def get_colors(f):
    id_img = np.array(f['frames']['0000']['images']['_id_cam0'][()]) # first frame, assumes all objects are in view
    id_img = np.array(Image.open(io.BytesIO(id_img))) # (256, 256, 3)
    colors = np.unique(id_img.reshape(-1, id_img.shape[2]), axis=0) # full list of unique colors in id map
    return colors

def build_labels(rois):
        assert rois.ndim == 3, rois.shape # (T, K, 4)
        assert rois.shape[-1] == 4, rois.shape
        pos = (rois[:,:,:2] + rois[:,:,2:] ) / 2 # get x,y pos
        off = pos[1:] - pos[:-1] # get x,y offset
        labels = np.concatenate([off, pos[1:]], axis=-1)
        return labels
    
def compute_bboxes_from_mask(id_img, colors, tol=0.1):
    bboxes = []
    for color in colors:
        if not color.any(): # all zeros
            continue
        idxs = np.argwhere(id_img==color)
        if len(idxs) > 0:
            x1 = np.min(idxs[:,1])
            x2 = np.max(idxs[:,1])
            y1 = np.min(idxs[:,0])
            y2 = np.max(idxs[:,0])
            xyxy = np.array([x1,y1,x2,y2])
            x,y,w,h = xyxy_to_xywh(*xyxy)
            w, h = w*(1+tol), h*(1+tol)
            xyxy = np.clip(xywh_to_xyxy(x,y,w,h), 0, id_img.shape[0]-1).astype(np.uint8)
        else:
            xyxy = -np.ones(4)
        bboxes.append(xyxy)
    return bboxes

def xyxy_to_xywh(x1, y1, x2, y2):
    x = (x1+x2)/2
    y = (y1+y2)/2
    w = (x2-x1)/2
    h = (y2-y1)/2
    return x,y,w,h

def xywh_to_xyxy(x, y, w, h):
    x1 = x - w
    x2 = x + w
    y1 = y - h
    y2 = y + h
    return x1,y1,x2,y2

class RPINDataset(Dataset):
    def __init__(
        self,
        data_root,
        indices=None,
        seq_len=25,
        state_len=7,
        subsample_factor=6,
        seed=0,
        ):
        videos = glob.glob(os.path.join(data_root, "**/*.hdf5"))
        corrupt = glob.glob(os.path.join(data_root, '**/temp.hdf5'))
        self.hdf5_files = list(set(videos) - set(corrupt))
        self.indices = indices
        self.seq_len = seq_len
        self.state_len = state_len # not necessarily always used
        assert self.seq_len > self.state_len, 'Sequence length {} must be greater than state length {}'.format(self.seq_len, self.state_len)
        self.subsample_factor = subsample_factor
        self.rng = np.random.RandomState(seed=seed)

        if self.indices is not None:
            logging.info('Dataset len: {}'.format(len(self.indices)))
        else:
            logging.info('Dataset len: {}'.format(len(self.hdf5_files)))            

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        else:
            return len(self.hdf5_files)   

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]
        with h5py.File(self.hdf5_files[index], 'r') as f: # load ith hdf5 file from list
            collision_ind, label, contact = get_label(self.hdf5_files[index])
            colors = get_colors(f)
            frames = list(f['frames'])
            target_contacted_zone = False
            for frame in reversed(frames):
                lbl = f['frames'][frame]['labels']['target_contacting_zone'][()]
                if lbl: # as long as one frame touching, label is True
                    target_contacted_zone = True
                    break

            images = []
            img_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                ])
            rois = []
            # object_ids = np.array(f['static']['object_ids'])
            prev_bboxes = 0
            for frame in frames[0:self.seq_len*self.subsample_factor:self.subsample_factor]:
                img = f['frames'][frame]['images']['_img_cam0'][()]
                if img.ndim == 1:
                    img = Image.open(io.BytesIO(img)) # (256, 256, 3)
                else:
                    img = Image.fromarray(img)
                img = img_transforms(img) # TODO: also need to rescale bboxes if resizing image
                images.append(img)
                if 'bboxes' in f['frames'][frame]:
                    bboxes = f['frames'][frame]['bboxes'][()]
                else:
                    id_img = f['frames'][frame]['images']['_id_cam0'][()]
                    id_img = np.array(Image.open(io.BytesIO(id_img))) # (256, 256, 3)
                    bboxes = compute_bboxes_from_mask(id_img, colors)
                bboxes, prev_bboxes = np.where(bboxes==-1, prev_bboxes, bboxes), bboxes
                rois.append(bboxes)
            if len(rois) < self.seq_len:
                images += [images[-1]] * (self.seq_len - len(rois))
                rois += [rois[-1]] * (self.seq_len - len(rois))

            rois = np.array(rois, dtype=np.float32)
            num_objs = rois.shape[1]
            if num_objs < 1:
                return self.__getitem__(index-1)
            max_objs = 10 # self.pretraining_cfg.MODEL.RPIN.NUM_OBJS # TODO: do padding elsewhere?
            #assert num_objs <= max_objs, f'num objs {num_objs} greater than max objs {max_objs}'
            ignore_mask = np.ones(max_objs, dtype=np.float32)
            if num_objs < max_objs:
                rois = np.pad(rois, [(0,0), (0, max_objs-num_objs), (0,0)])
                ignore_mask[num_objs:] = 0
            labels = torch.from_numpy(build_labels(rois)[self.state_len-1:])
            rois = torch.from_numpy(rois)
            images = torch.stack(images, dim=0)
            binary_labels = torch.ones((self.seq_len, 1)) if target_contacted_zone else torch.zeros((self.seq_len, 1)) # Get single label over whole sequence
            stimulus_name = f['static']['stimulus_name'][()]

        sample = {
            'data': images[:15],#self.state_len],
            'rois': rois[:, :max_objs],
            'labels': labels[:, :max_objs], # [off, pos]
            'data_last': images[:self.state_len],
            'ignore_mask': torch.from_numpy(ignore_mask),
            'stimulus_name': stimulus_name,
            'binary_labels': binary_labels[0],
            'images': images,
            'contacts': contact,
            'collision_ind': collision_ind,
        }
        return sample
    
class RPINTrainDataset(Dataset):
    def __init__(self, data_path, indices = None):
        self.dataset = h5py.File(data_path, 'r')  
        self.indices = indices

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return self.dataset['data'].shape[0]   

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]
        data = self.dataset["data"][index]
        boxes = self.dataset["rois"][index]
        labels = self.dataset["labels"][index]
        data_last = self.dataset["data_last"][index]
        ignore_mask = self.dataset["ignore_mask"][index]
        binary_label = self.dataset["binary_labels"][index]
        return data[:5], boxes[:16, :10, :], labels[:11, :10, :], data_last[:5], ignore_mask[:10], data[:15], binary_label
