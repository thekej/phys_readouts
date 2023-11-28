import os
import cv2
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import glob
import gzip
import pickle
import h5py
import csv

import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from data_new import load_data, load_data_dominoes, prepare_input, normalize, denormalize, recalculate_velocities, \
                 correct_bad_chair, remove_large_obstacles, subsample_particles_on_large_objects
from models import SGNN
from utils import mkdir, get_query_dir
from utils_geom import calc_rigid_transform

training_data_root = get_query_dir("training_data_dir")
testing_data_root = get_query_dir("testing_data_dir")

data_root = get_query_dir("dpi_data_dir")
model_root = get_query_dir("out_dir")
out_root = os.path.join(model_root, "eval")
random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--env', default='')
parser.add_argument('--time_step', type=int, default=250)
parser.add_argument('--time_step_clip', type=int, default=0)
parser.add_argument('--dt', type=float, default=1./60.)
parser.add_argument('--training_fpt', type=float, default=1)
parser.add_argument('--subsample', type=int, default=3000)

# parser.add_argument('--nf_relation', type=int, default=300)
# parser.add_argument('--nf_particle', type=int, default=200)
# parser.add_argument('--nf_effect', type=int, default=200)
parser.add_argument('--n_layer', type=int, default=3)
parser.add_argument('--p_step', type=int, default=4)
parser.add_argument('--hidden_dim', type=int, default=512)

parser.add_argument('--rand_rot', type=int, default=0)
parser.add_argument('--pred_only', type=int, default=1)

parser.add_argument('--modelf', default='files')
parser.add_argument('--dataf', default='data')
parser.add_argument('--evalf', default='eval')
parser.add_argument('--mode', default='valid')
parser.add_argument('--statf', default="")
parser.add_argument('--eval', type=int, default=1)
parser.add_argument('--gt_only', type=int, default=0)
parser.add_argument('--test_training_data_processing', type=int, default=0)
parser.add_argument('--ransac_on_pred', type=int, default=0)
parser.add_argument('--verbose_data', type=int, default=0)
parser.add_argument('--verbose_model', type=int, default=0)
parser.add_argument('--model_name', default='DPINet2')

parser.add_argument('--debug', type=int, default=0)

parser.add_argument('--n_instances', type=int, default=0)
parser.add_argument('--n_stages', type=int, default=0)
parser.add_argument('--n_his', type=int, default=0)
# shape state:
# [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
parser.add_argument('--shape_state_dim', type=int, default=14)

# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)
parser.add_argument('--position_dim', type=int, default=0)

# relation attr:
parser.add_argument('--relation_dim', type=int, default=0)

#visualization
parser.add_argument('--interactive', type=int, default=0)
parser.add_argument('--saveavi', type=int, default=0)
parser.add_argument('--save_pred', type=int, default=1)
parser.add_argument('--save_file_ocp', type=str)
parser.add_argument('--save_file_ocd', type=str)
parser.add_argument('--save_file_focused', type=str)
parser.add_argument('--save_file_sim', type=str)

args = parser.parse_args()

phases_dict = dict()

if args.env == "TDWdominoes":
    args.n_rollout = 2# how many data
    data_names = ['positions', 'velocities']
    args.time_step = 200
    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [rigid, fluid, root_0]
    args.attr_dim = 3
    args.dt = 0.01

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = -1
    args.time_step_clip = 0
    args.n_stages = 4
    args.n_stages_types = ["leaf-leaf", "leaf-root", "root-root", "root-leaf"]

    args.neighbor_radius = 0.08
    args.gen_data = False

    phases_dict = dict()  # load from data
    model_name = copy.deepcopy(args.modelf)
    #args.modelf = 'dump/' + args.modelf
    #args.modelf = os.path.join(model_root, args.modelf)
else:
    raise AssertionError("Unsupported env")


gt_only = args.gt_only

evalf_root = os.path.join(out_root, args.evalf + '_' + args.env, model_name)
mkdir(os.path.join(out_root, args.evalf + '_' + args.env))
mkdir(evalf_root)

mode = args.mode
if mode == "train":
    hdf5_root = training_data_root
elif mode == "test":
    hdf5_root = testing_data_root
else:
    raise ValueError
data_root_ori = data_root
scenario = args.dataf
args.data_root = data_root

prefix = args.dataf
if gt_only:
    prefix += "_gtonly"

args.dataf = os.path.join(data_root, mode, args.dataf)
stat = [np.zeros((3, 3)), np.zeros((3, 3))]

if not gt_only:
    if args.statf:
        stat_path = os.path.join(data_root_ori, args.statf)
        print("Loading stored stat from %s" % stat_path)
        stat = load_data(data_names[:2], stat_path)
        for i in range(len(stat)):
            stat[i] = stat[i][-args.position_dim:, :]
            # print(data_names[i], stat[i].shape)

    use_gpu = torch.cuda.is_available()

    if args.model_name == 'SGNN':
        args.noise_std = 3e-4
        model = SGNN(n_layer=args.n_layer, s_dim=4, hidden_dim=args.hidden_dim, activation=nn.SiLU(),
                     cutoff=0.08, gravity_axis=1, p_step=args.p_step)
    else:
        raise ValueError(f"no such model {args.model_name} for env {args.env}" )

    if args.epoch == 0 and args.iter == 0:
        model_file = os.path.join(args.modelf, 'net_best.pth')
    else:
        model_file = os.path.join(args.modelf, 'net_epoch_%d_iter_%d.pth' % (args.epoch, args.iter))

    print("Loading network from %s" % model_file)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    criterionMSE = nn.MSELoss()

    if use_gpu:
        model.cuda()

mode = args.mode

# list all the args
# only evaluate on human data now

infos = np.arange(100)
data_name = args.dataf.split("/")[-1]

if args.save_pred:

    pred_gif_folder = os.path.join(evalf_root, mode + "-"+ scenario)

    if args.ransac_on_pred:
        pred_gif_folder = os.path.join(evalf_root, "ransacOnPred-" + mode +  scenario)
    mkdir(pred_gif_folder)

dt = args.training_fpt * args.dt

#gt_preds = []
#arg_names = [file for file in os.listdir(args.dataf) if not file.endswith("txt")]
trial_full_paths = []
#for arg_name in arg_names:
#    trial_full_paths.append(os.path.join(args.dataf, arg_name))
data_root = '/ccn2/u/thekej/sgnn_readout/test/data_balanced/'
trial_full_paths = glob.glob(os.path.join(data_root, "*/*"))
#trial_full_paths = list(os.walk(data_root))
trial_full_paths = trial_full_paths#[3000:3010]
#trial_full_paths = random.sample(trial_full_paths, 50)

# set up new dataset
ocp, ocd, ocd_focused = [], [], []
labels = []
contacts = []
filenames = []
stimulus_map = {}
all_scenarios = {'collide': [], 'drop': [], 'support': [], 'link': [], 'roll': [], 'contain': [],
                         'dominoes': []}

for trial_id, trial_name in enumerate(trial_full_paths):
    print('processing case', trial_id, 'total', len(trial_full_paths))
    #scenarios = 'drop'
    stimulus_map[trial_name] = trial_id
    filenames += [trial_name]
    for key in all_scenarios.keys():
        if key in trial_name:
            all_scenarios[key].append(trial_id)
            
    gt_node_rs_idxs = []
    args.time_step = len([file for file in os.listdir(trial_name) if file.endswith(".h5")]) - 1

    print("Rollout %d / %d" % (trial_id, len(trial_full_paths)))

    timesteps = [t for t in range(0, args.time_step - int(args.training_fpt), int(args.training_fpt))]
    #if len(timesteps) > 150:
    #    timesteps = [t for t in range(0, 150, int(args.training_fpt))]
    max_timestep = len(timesteps)

    if args.env == "TDWdominoes":
        pkl_path = os.path.join(trial_name, 'phases_dict.pkl')
        with open(pkl_path, "rb") as f:
            phases_dict = pickle.load(f)

    phases_dict["trial_dir"] = trial_name
    labels += [phases_dict["label"]]
    contacts += [phases_dict["contact"]]
    frame_label = phases_dict["collision_ind"]

    if args.test_training_data_processing:
        is_bad_chair = correct_bad_chair(phases_dict)
        # remove obstacles that are too big
        is_remove_obstacles = remove_large_obstacles(phases_dict)
        # downsample large object
        is_subsample = subsample_particles_on_large_objects(phases_dict, limit=args.subsample)
        # is_subsample = True
        print("trial_id", trial_id, "is_bad_chair", is_bad_chair, "is_remove_obstacles", is_remove_obstacles, "is_subsample", is_subsample)
        print("trial_name", trial_name)
    else:
        is_bad_chair = correct_bad_chair(phases_dict)
        # remove obstacles that are too big
        is_remove_obstacles = remove_large_obstacles(phases_dict, limit=args.subsample)
        # downsample large object
        # is_subsample = subsample_particles_on_large_objects(phases_dict, 4000)

    if args.rand_rot:
        print('Using Random Rotation!')
        temp_data_path = os.path.join(trial_name, str(0) + '.h5')
        temp_data = load_data_dominoes(data_names, temp_data_path, phases_dict)
        center = np.mean(temp_data[0], axis=0).reshape(1, -1)
        d = np.zeros(3)
        d[1] = 1
        x, y, z = d[0], d[1], d[2]
        theta = 0.5 * np.pi
        # theta = (np.random.rand(1) - 0.5) * 4
        cos, sin = np.cos(theta), np.sin(theta)
        ret = [
            [cos + (1 - cos) * x * x,
             (1 - cos) * x * y - sin * z,
             (1 - cos) * x * z + sin * y],
            [(1 - cos) * x * y + sin * z,
             cos + (1 - cos) * y * y,
             (1 - cos) * y * z - sin * x],
            [(1 - cos) * x * z - sin * y,
             (1 - cos) * y * z + sin * x,
             cos + (1 - cos) * z * z]
        ]
        Q = np.array(ret)
        
    n_actual_frames = len(timesteps)
    
    # used for training heuristic on  detection
    indices_ocd_focussed = [frame_label]

    indices_ocd = np.arange(1, n_actual_frames, 40 // 10)
    

    if 'collide' in trial_name:
        max_frame = 15
    else:
        max_frame = 45

    indices_ocp = np.arange(max_frame, 0, -40 // 10).clip(0, n_actual_frames - 1)[::-1]
    if len(indices_ocp) < 12:
        indices_ocp = np.concatenate([np.array([indices_ocp[0]] * (12 - len(indices_ocp))), 
                                      indices_ocp])
        
    indices_sim = np.arange(max_frame, 225, 40//10).clip(1, n_actual_frames - 1)


    ocp_entry, ocd_entry, focus_entry, simulation = [], [], [], []
    for entry, start_timestep in enumerate(range(1, n_actual_frames)):
        
        ocp_flag, ocd_flag, focus_flag = False, False, False
        if entry in indices_ocp:
            ocp_flag = True
        if entry in indices_ocd:
            ocd_flag = True
        if entry in indices_ocd_focussed:
            focus_flag = True
            
        if not ocp_flag and not ocd_flag and not focus_flag:
            continue
        
        # model rollout
        start_id = 1  # 5
        if n_actual_frames < start_timestep:
            start_timestep = n_actual_frames - 1
        data_path = os.path.join(trial_name, f'{start_timestep}.h5')

        data = load_data_dominoes(data_names, data_path, phases_dict)
        if args.rand_rot:
            data[0] = np.matmul((data[0] - center), Q) + center
        data_path_prev = os.path.join(trial_name, f'{int(start_timestep - args.training_fpt)}.h5')
        data_prev = load_data_dominoes(data_names, data_path_prev, phases_dict)
        if args.rand_rot:
            data_prev[0] = np.matmul((data_prev[0] - center), Q) + center
        _, data = recalculate_velocities([data_prev, data], dt, data_names)


        x = data[0]
        v = data[1]
        h = np.zeros((x.shape[0], 4))
        obj_id = np.zeros_like(x)[..., 0]
        obj_type = []
        instance_idx = phases_dict['instance_idx']
        for i in range(len(instance_idx) - 1):
            obj_id[instance_idx[i]:instance_idx[i + 1]] = i
            obj_type.append(phases_dict['material'][i])
            if phases_dict['material'][i] == 'rigid':
                h[instance_idx[i]:instance_idx[i + 1], 0] = 1
            elif phases_dict['material'][i] in ['cloth', 'fluid']:
                h[instance_idx[i]:instance_idx[i + 1], 1] = 1
        x = torch.Tensor(x)
        v = torch.Tensor(v)
        h = torch.Tensor(h)
        obj_id = torch.LongTensor(obj_id)

        buf = [x, v, h, obj_id]

        with torch.set_grad_enabled(False):
            for d in range(len(buf)):
                if type(buf[d]) == list:
                    for t in range(len(buf[d])):
                        buf[d][t] = Variable(buf[d][t].cuda())
                else:
                    buf[d] = Variable(buf[d].cuda())

            x, v, h, obj_id = buf
            feats = model.extract_objects(x, v, h, obj_id, obj_type, phases_dict['yellow_id'], phases_dict['red_id'])
            #feats = model.extract(x, v, h, obj_id, obj_type)
            if ocp_flag:
                ocp_entry += [feats.cpu().numpy()]
            if ocd_flag:
                ocd_entry += [feats.cpu().numpy()]
            if focus_flag:
                focus_entry += [feats.cpu().numpy()]
            
    if len(ocp_entry) < 12:
        ocp_entry = np.concatenate([np.array([ocp_entry[0]] * (12 - len(ocp_entry))),
                                               ocp_entry])

    ocp += [np.stack(ocp_entry)]
    ocd += [np.stack(ocd_entry)]
    ocd_focused += [np.stack(focus_entry)]
    
    sim = ocp_entry
    
    for current_fid in range(45, min(n_actual_frames, 225)):
        x = data[0]
        v = data[1]
        h = np.zeros((x.shape[0], 4))
        obj_id = np.zeros_like(x)[..., 0]
        obj_type = []
        instance_idx = phases_dict['instance_idx']
        for i in range(len(instance_idx) - 1):
            obj_id[instance_idx[i]:instance_idx[i + 1]] = i
            obj_type.append(phases_dict['material'][i])
            if phases_dict['material'][i] == 'rigid':
                h[instance_idx[i]:instance_idx[i + 1], 0] = 1
            elif phases_dict['material'][i] in ['cloth', 'fluid']:
                h[instance_idx[i]:instance_idx[i + 1], 1] = 1
        x = torch.Tensor(x)
        v = torch.Tensor(v)
        h = torch.Tensor(h)
        obj_id = torch.LongTensor(obj_id)

        buf = [x, v, h, obj_id]

        with torch.set_grad_enabled(False):
            if use_gpu:
                for d in range(len(buf)):
                    if type(buf[d]) == list:
                        for t in range(len(buf[d])):
                            buf[d][t] = Variable(buf[d][t].cuda())
                    else:
                        buf[d] = Variable(buf[d].cuda())
            else:
                for d in range(len(buf)):
                    if type(buf[d]) == list:
                        for t in range(len(buf[d])):
                            buf[d][t] = Variable(buf[d][t])
                    else:
                        buf[d] = Variable(buf[d])
            x, v, h, obj_id = buf

            vels, feats = model(x, v, h, obj_id, obj_type, phases_dict['yellow_id'], phases_dict['red_id'])
            if current_fid in indices_sim:
                sim += [feats.cpu().numpy()]
                
        vels = denormalize([vels.data.cpu().numpy()], [stat[1]])[0]
        if args.ransac_on_pred:
            positions_prev = data[0]
            predicted_positions = data[0] + vels * dt
            for obj_id in range(len(instance_idx) - 1):
                st, ed = instance_idx[obj_id], instance_idx[obj_id + 1]
                if phases_dict['material'][obj_id] == 'rigid':

                    pos_prev = positions_prev[st:ed]
                    pos_pred = predicted_positions[st:ed]

                    R, T = calc_rigid_transform(pos_prev, pos_pred)
                    refined_pos = (np.dot(R, pos_prev.T) + T).T

                    predicted_positions[st:ed, :] = refined_pos
            data[0] = predicted_positions
            data[1] = (predicted_positions - positions_prev)/dt
        else:
            data[0] = data[0] + vels * dt
            data[1][:, :args.position_dim] = vels

    simulation += [np.stack(sim)]
    
max_dim = max([x.shape[0] for x in ocd])

for ct, f in enumerate(ocd):
    # Get the number of dimensions
    num_dims = f.ndim
    # Create a padding configuration that respects the tensor's dimensions
    # The padding is zero for all dimensions except for the second one.
    padding = [(0, 0)] * num_dims
    padding[0] = (0, max_dim - f.shape[0])  # Pad the second dimension

    # Apply padding and update the list item
    ocd[ct] = np.pad(f, padding, mode='constant')

dt = h5py.special_dtype(vlen=str)

print('save 1')
with h5py.File(args.save_file_ocp,'w') as hf:
    hf.create_dataset("features", data=np.stack(ocp))
    hf.create_dataset("label", data=np.array(labels))
    hf.create_dataset("contacts", data=np.concatenate(contacts))
    hf.create_dataset("filenames", data=filenames, dtype=dt)
    
import json
with open('/ccn2/u/thekej/models/sgnn_physion/ocp/test_json.json', 'w') as f:
    json.dump(stimulus_map, f)
    
with open('/ccn2/u/thekej/models/sgnn_physion/ocp/test_scenario_map.json', 'w') as f:
    json.dump(all_scenarios, f)

print('save 2')
with h5py.File(args.save_file_ocd ,'w') as hf:
    hf.create_dataset("features", data=np.stack(ocd))
    hf.create_dataset("label", data=np.array(labels))
    hf.create_dataset("contacts", data=np.concatenate(contacts))
    hf.create_dataset("filenames", data=filenames, dtype=dt)
    
with open('/ccn2/u/thekej/models/sgnn_physion/ocd/test_json.json', 'w') as f:
    json.dump(stimulus_map, f)
    
with open('/ccn2/u/thekej/models/sgnn_physion/ocd/test_scenario_map.json', 'w') as f:
    json.dump(all_scenarios, f)

print('save 3')
with h5py.File(args.save_file_focused ,'w') as hf:
    hf.create_dataset("features", data=np.stack(ocd_focused))
    hf.create_dataset("label", data=np.array(labels))
    hf.create_dataset("contacts", data=np.concatenate(contacts))
    hf.create_dataset("filenames", data=filenames, dtype=dt)

with open('/ccn2/u/thekej/models/sgnn_physion/ocd_focussed/test_json.json', 'w') as f:
    json.dump(stimulus_map, f)
    
with open('/ccn2/u/thekej/models/sgnn_physion/ocd_focussed/test_scenario_map.json', 'w') as f:
    json.dump(all_scenarios, f)

print('save 4')
with h5py.File(args.save_file_sim ,'w') as hf:
    hf.create_dataset("features", data=np.stack(simulation))
    hf.create_dataset("label", data=np.array(labels))
    hf.create_dataset("contacts", data=np.concatenate(contacts))
    hf.create_dataset("filenames", data=filenames, dtype=dt)

with open('/ccn2/u/thekej/models/sgnn_physion/sim/test_json.json', 'w') as f:
    json.dump(stimulus_map, f)
    
with open('/ccn2/u/thekej/models/sgnn_physion/sim/test_scenario_map.json', 'w') as f:
    json.dump(all_scenarios, f)
    
