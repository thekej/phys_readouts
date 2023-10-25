
import glob
import h5py
import io

from PIL import Image
from torch.utils.data import Dataset


buggy_stims = "pilot-containment-cone-plate_0017 \
pilot-containment-cone-plate_0022 \
pilot-containment-cone-plate_0029 \
pilot-containment-cone-plate_0034 \
pilot-containment-multi-bowl_0042 \
pilot-containment-multi-bowl_0048 \
pilot-containment-vase_torus_0031 \
pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom_0005 \
pilot_it2_collision_non-sphere_box_0002 \
pilot_it2_collision_non-sphere_tdw_1_dis_1_occ_0004 \
pilot_it2_collision_non-sphere_tdw_1_dis_1_occ_0007 \
pilot_it2_drop_simple_box_0000 \
pilot_it2_drop_simple_box_0042 \
pilot_it2_drop_simple_tdw_1_dis_1_occ_0003 \
pilot_it2_rollingSliding_simple_collision_box_0008 \
pilot_it2_rollingSliding_simple_collision_box_large_force_0009 \
pilot_it2_rollingSliding_simple_collision_tdw_1_dis_1_occ_0002 \
pilot_it2_rollingSliding_simple_ledge_tdw_1_dis_1_occ_sphere_small_zone_0022 \
pilot_it2_rollingSliding_simple_ramp_box_small_zone_0006 \
pilot_it2_rollingSliding_simple_ramp_tdw_1_dis_1_occ_small_zone_0004 \
pilot_it2_rollingSliding_simple_ramp_tdw_1_dis_1_occ_small_zone_0017 \
pilot_linking_nl1-8_mg000_aCyl_bCyl_tdwroom1_long_a_0022 \
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom1_0012 \
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0006 \
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0010 \
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0029 \
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0036 \
pilot_linking_nl6_aNone_bCone_occ1_dis1_boxroom_0028 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0000 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0002 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0003 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0010 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0013 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0017 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0018 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0032 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0036 \
pilot_towers_nb4_fr015_SJ000_gr01_mono0_dis1_occ1_tdwroom_unstable_0021 \
pilot_towers_nb4_fr015_SJ000_gr01_mono0_dis1_occ1_tdwroom_unstable_0041 \
pilot_towers_nb5_fr015_SJ030_mono0_dis0_occ0_boxroom_unstable_0006 \
pilot_towers_nb5_fr015_SJ030_mono0_dis0_occ0_boxroom_unstable_0009".split(' ')


import numpy as np
def get_label(f):
#     try:
    with h5py.File(f) as h5file:

        for key in h5file['frames'].keys():
            lbl = np.array(h5file['frames'][key]['labels']['target_contacting_zone']).item()
            if lbl:
                return int(key), 1# True

        ind = len(h5file['frames'].keys()) // 2

        return ind, 0# False

class UnifiedPhysion(Dataset):

    def __init__(self, hdf5_path, 
                 frame_duration=150, 
                 ocd=False,
                 video_len = 25,
                 n_context = 13
                 ):

        if 'test_consolidated' in hdf5_path:
            self.buggy_stims = buggy_stims
        else:
            self.buggy_stims = []

        self.all_hdf5 = glob.glob(hdf5_path + '/*/*.hdf5')

        self.all_hdf5 = [x for x in self.all_hdf5 if ('temp' not in x) and ('encoding' not in x)]

        blacklisted_inds = []

        for ct, f in enumerate(self.all_hdf5):
            if str(f).split('/')[-1].split('.')[0] not in self.buggy_stims:
                blacklisted_inds.append(f)

        self.all_hdf5 = blacklisted_inds

        self.get_label = get_label
        
        self.frame_duration = frame_duration
        
        self.frame_gap = frame_duration / 10
        
        self.ocd = ocd
        
        self.video_len = video_len
        
        self.n_context = n_context

    def __len__(self):
        return len(self.all_hdf5)


    def __getitem__(self, idx):
        filename = self.all_hdf5[idx]
        with h5py.File(filename, 'r') as h5_file:

            op = get_label(filename)
            frame_label = op[0]

            frame_label = (frame_label - (frame_label % self.frame_gap))
            ret = {}
            ret['label'] = op[1]
            ret['frame_label'] = frame_label
            frames = list(h5_file['frames'])
            
            if self.ocd:
                window = self.n_context / 2
                indices = np.arange(frame_label - (window - 1)*self.frame_gap, 
                                    frame_label + window*self.frame_gap, 
                                    self.frame_gap).clip(0, len(frames) - 1)
            else:
                indices = np.arange(0, len(frames), self.frame_gap)
                
                if 'collide' in filename:
                    # print("file is collide", filename)
                    index_start = int(300 / self.frame_duration)
                    for i in range(0, indices.shape[0])[::-1]:
                        if i > index_Start:
                            indices[i] = indices[i - index_start]
                        else:
                            indices[i] = 0

            indices = indices.tolist()

            

            images = []
            contact_check = False
            contact_points = np.array([[-1, -1]])
            for i, frame in enumerate(frames):
                if not i in indices:
                    continue
                img = h5_file['frames'][frame]['images']['_img_cam0'][:]
                img = Image.open(io.BytesIO(img)) # (256, 256, 3)
                images.append(img)
                
                if len(images) >= self.video_len:
                    break
            
                contacts = h5_file['frames'][frame]['collisions']['contacts_ot']
                if contacts.shape[0] == 0:
                    continue
                if not contact_check:
                    contact_points = get_contact(h5_file, frame)
                    contact_check = True
                
                    
            if len(images) < self.video_len:
                    images += [images[-1]] * (self.video_len - len(images))
                
            ret['contacts'] = contact_points

        ret['video'] = np.stack(images)
        ret['filename'] = filename

        return ret
    
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
