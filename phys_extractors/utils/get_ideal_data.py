import h5py
import scipy.io as sio
import pytorch3d.transforms
import torch
import numpy as np

def decode_numpy(centroid, euler_angles, size, pts):
    '''

    :param centroid: 3,
    :param euler_angles: 3,
    :param size: 3,
    :param pts: [N, 3] np array
    :return:
    '''
    
    mat = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor(euler_angles), 'XYZ').numpy()
        # print(euler_angles)

    all_pts = []
    for i, pt in enumerate(pts):
        p1 = centroid + np.matmul(mat.T, size * pt)
        all_pts.append(p1)
        
    all_pts = np.stack(all_pts, 0)

    return all_pts


path = '/ccn2/u/rmvenkat/data/final_set_cvpr_results_fixed_bugs/gt_3d_with_full_250/ocd/'

mat = sio.loadmat('/ccn2/u/rmvenkat/data/dict_pts.mat')

mat = {k.encode(): v for k, v in mat.items()}

print(mat.keys())

#map keys to ind
dict_key_to_ind = {}
dict_ind_to_key = {}
for ct, k in enumerate(mat.keys()):
    dict_key_to_ind[k] = ct
    dict_ind_to_key[ct] = k

new_file_path = '/ccn2/u/thekej/models/3d_point_cloud/ocd/'

for file in ['test_features.hdf5', 'train_features.hdf5']:
    with h5py.File(path + file, 'r') as original_file:
        data = original_file['features'] #(1035, 801, 11, 14)
        data = data[:, :225]
        N, T, obj, _ = data.shape
        data = data.reshape(-1, 14)
        data_decoded = []
        for i, d in enumerate(data):
            if i % 10000 == 0:
                print(file,i, 'out of ', data.shape[0])
            if d[10] == 0:
                data_decoded += [np.zeros((2048, 3))]
            else:
                centroid = d[:3]
                euler_angle = d[3:6]
                scale = d[6:9]
                pts = dict_ind_to_key[d[10]]
                pts_decoded = decode_numpy(centroid, euler_angle, scale, mat[pts])
                data_decoded += [pts_decoded]
    
        data_decoded = np.stack(data_decoded)
        data_decoded = data_decoded.reshape(N, T, obj, 2048, 3)
        # Generate a tensor where each T slice has its corresponding value of T (ranging from 0 to T-1)
        t_values = torch.arange(T).view(1, T, 1, 1, 1).expand(N, T, obj, 2048, 1).float()
    
        # Concatenate the original tensor with this new tensor along the last dimension
        print(t_values.shape, data_decoded.shape)
        data_decoded = torch.cat((torch.tensor(data_decoded), t_values), dim=-1)
    
        with h5py.File(new_file_path+file, 'w') as new_file:
            # Copy all keys except for 'features' from the original to the new file
            for key in original_file.keys():
                if key != 'features':
                    original_file.copy(key, new_file)
            
            # Convert `data_decoded` to numpy if it's a torch tensor
            if isinstance(data_decoded, torch.Tensor):
                data_decoded_numpy = data_decoded.numpy()
            else:
                data_decoded_numpy = data_decoded
            
            # Write the modified 'features' data to the new file
            new_file.create_dataset('features', data=data_decoded_numpy)