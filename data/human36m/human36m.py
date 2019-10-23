"""
Code Human 3.6M PyTorch Dataloader.
Dataset link: http://vision.imar.ro/human3.6m/description.php 
Preprocessing code: https://github.com/anibali/h36m-fetch

Notes:
*** This file requires another piece of code `skeleton.py`.
"""

import os
import h5py
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from functools import reduce

from skeleton import Skeleton
osp = os.path


STD_SCALE = 3


class Human36mDataset(Dataset):
    def __init__(self, data_root, max_seq_len, delta_len, n_breakpoints=0, speed_range=[1, 1], acc_range=[-1, 1], train=True, remove_static_joints=True, mode='train'):
        # Set arguments
        self.data_root = osp.abspath(osp.expanduser(data_root))
        self.max_seq_len = max_seq_len
        self.delta_len = delta_len
        self.speed_range = speed_range
        self.n_breakpoints = n_breakpoints
        self.acc_range = acc_range
        self.remove_static_joints = remove_static_joints
        self.train = train
        self.subseq_len = self.delta_len * 2
        self.skeleton = Skeleton(parents=[-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
                                          16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                                 joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                 joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
        assert mode in ['train', 'test'], 'Invalid mode for Human36mDataset. Must be \'train\' or \'test\'.'
        self.mode = mode

        # Read dataset
        self.raw_data = read_human36m(self.data_root, self.mode)

        # Reformat data
        self.data = reformat_data(self.raw_data)

        # Filter out sequence that is shorter than `max_seq_len`
        self.data['pose']['3d'] = list(filter(lambda x: x.shape[0] >= self.max_seq_len, self.data['pose']['3d']))
        self.data['pose']['2d'] = list(filter(lambda x: x.shape[0] >= self.max_seq_len, self.data['pose']['2d']))

        # Remove static joints
        if self.remove_static_joints:
            # Bring the skeleton to 17 joints instead of the original 32
            self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
            
            # Rewire shoulders to the correct parents
            self.skeleton._parents[11] = 8
            self.skeleton._parents[14] = 8

        # Preprocess data; NOTE: make sure index of pivot joint is correct, if used.
        align_and_normalize_dataset_v2(self.data)
        
    def __getitem__(self, idx):
        # Fetch data
        pose_2d = self.data['pose']['2d'][idx]
        pose_3d = self.data['pose']['3d'][idx]
        camera_view = self.data['camera_view'][idx]

        # Crop a sub-sequence
        total_len = pose_3d.shape[0]
        start = np.random.randint(low=0, high=total_len-self.speed_range[1]*self.max_seq_len+1)
        pose_2d_cropped, pose_3d_cropped, speed = [], [], []
        if self.n_breakpoints > 0: # Varying video speed
            start_offset = 5
            bps = [0, self.max_seq_len] + list(np.random.randint(low=1+start_offset, high=self.max_seq_len-start_offset, size=self.n_breakpoints))
            bps = sorted(bps)
            speed.append(np.random.randint(low=self.speed_range[0], high=self.speed_range[1]+1))
            for bp_i, bp in enumerate(bps[1:]):
                end = start + (bp - bps[bp_i])*speed[-1]
                pose_2d_cropped.append(pose_2d[start:end:speed[-1]].copy())
                pose_3d_cropped.append(pose_3d[start:end:speed[-1]].copy())
                next_speed = min(max(speed[-1] + np.random.randint(low=self.acc_range[0], high=self.acc_range[1]+1), 1), self.speed_range[1])
                speed.append(next_speed)
                start = end
            speed = speed[:-1] # The last speed not used
        else: # Constant video speed
            bps = []
            speed = np.random.randint(low=self.speed_range[0], high=self.speed_range[1]+1)
            pose_2d_cropped.append(pose_2d[start:start+self.max_seq_len*speed:speed].copy())
            pose_3d_cropped.append(pose_3d[start:start+self.max_seq_len*speed:speed].copy())
        pose_2d_cropped = np.concatenate(pose_2d_cropped, 0)
        pose_3d_cropped = np.concatenate(pose_3d_cropped, 0)

        # Pack data
        data = {
            'pose_2d': pose_2d_cropped,
            'pose_3d': pose_3d_cropped,
            'camera_view': camera_view,
            'speed': speed,
            'breakpoints': bps
        }

        return data

    def get_seq_len(self):
        seq_len = np.random.randint(low=self.max_seq_len-2*self.delta_len, high=self.max_seq_len+1)
        return seq_len

    def remove_joints(self, joints_to_remove):
        kept_joints = self.skeleton.remove_joints(joints_to_remove)
        for seq_i in range(len(self.data['pose']['3d'])):
            self.data['pose']['3d'][seq_i] = self.data['pose']['3d'][seq_i][:, kept_joints]
            self.data['pose']['2d'][seq_i] = self.data['pose']['2d'][seq_i][:, kept_joints]

    def __len__(self):
        return len(self.data['pose']['2d'])


def read_human36m(root_dir, mode='train'):
    """ Read Human3.6M dataset. 
        Data structure of `data_dict`:
            k`annot`:
                list: each element is the meta-data of a sequence.
                    k`path` (str): full path to the .h5 file.
                    k`pose`:
                        k`2d`: 2D pose projected to the image; Shape = [N(sequence length) x 4(camera views), 32, 2].
                        k`3d`: 3D pose; Shape = [N x 4(camera views), 32, 3].
                        k`3d-univ`: NOTE: not sure what's the difference from k`3d`
                    k`action`, k`camera`, k`frame`, k`intrinsics`, 
                        k`intrinsics-univ`, `subaction`, `subject`: not used.
    """
    mode_id = ['S1', 'S5', 'S6', 'S7', 'S8'] if mode == 'train' else ['S9', 'S11']
    data_dict = {'annot': []}
    # Iterate through all identities (S1, S2, ...)
    id_names = sorted(os.listdir(root_dir))
    for id_i, id_n in enumerate(id_names):
        if id_n not in mode_id:
            continue
        # Iterate through all actions for a specific identity
        act_names = sorted(os.listdir(osp.join(root_dir, id_n)))
        for act_i, act_n in enumerate(act_names):
            # Directory that contains skeleton data and image data
            act_dir = osp.join(root_dir, id_n, act_n)

            # Read skeleton data (saved in the same .h5 file for the entire sequence)
            annot_path = osp.join(act_dir, 'annot.h5')
            if osp.exists(annot_path):
                h5_data = h5py.File(annot_path, 'r')
                annot = {'path': annot_path}
                for k, v in h5_data.items():
                    if isinstance(v, h5py.Group):
                        annot[k] = dict()
                        for k_, v_ in v.items():
                            annot[k][k_] = np.array(v_)
                    else:
                        annot[k] = np.array(v)
                data_dict['annot'].append(annot)
            else:
                print('[WARNING] {} does not exist!!!!!!!!!!!!!!!!!!!!!'.format(annot_path))

    return data_dict


def reformat_data(raw_data):
    """ Reformat the raw data for better usage in our dataloader. """
    def get_1view_data(x):
        N = x.shape[0] // 4
        return [x[:N]]
    def get_4view_data(x):
        N = x.shape[0] // 4
        return x[:N], x[N:2*N], x[2*N:3*N], x[3*N:4*N]
    get_view_data = get_1view_data

    data = {'pose': {'2d': [], '3d': []}, 'camera_view': []}
    for i in range(len(raw_data['annot'])):
        data['pose']['2d'].extend(get_view_data(raw_data['annot'][i]['pose']['2d']))
        data['pose']['3d'].extend(get_view_data(raw_data['annot'][i]['pose']['3d']))
        data['camera_view'].extend([0, 1, 2, 3])
    
    return data


def align_and_normalize_dataset(data):
    """ Align the entire dataset with respect to pivot joint. """
    norm_mode = ['axis_norm', 'all_norm'][0]
    n_data = len(data['pose']['2d'])
    # Align all skeletons in the dataset with respect to pivot joint
    pivot = 11 # NOTE: before removing static joints
    box2d_x, box2d_y = [], []
    box3d_x, box3d_y, box3d_z = [], [], []
    for i in range(n_data):
        # Perform alignment
        seq_2d = data['pose']['2d'][i]
        seq_3d = data['pose']['3d'][i]
        
        pivot_2d = seq_2d[:, pivot:pivot+1, :]
        pivot_3d = seq_3d[:, pivot:pivot+1, :]

        data['pose']['2d'][i] -= pivot_2d
        data['pose']['3d'][i] -= pivot_3d

        # Compute the size of bounding box; Note that x,y,z here do not correspond to those in other part of this code
        box2d_x.append([seq_2d[:,:,0].min(), seq_2d[:,:,0].max()])
        box2d_y.append([seq_2d[:,:,1].min(), seq_2d[:,:,1].max()])
        box3d_x.append([seq_3d[:,:,0].min(), seq_3d[:,:,0].max()])
        box3d_y.append([seq_3d[:,:,1].min(), seq_3d[:,:,1].max()])
        box3d_z.append([seq_3d[:,:,2].min(), seq_3d[:,:,2].max()])

    # Normalize skeletons to [0,1]
    box2d_x, box2d_y = np.array(box2d_x), np.array(box2d_y)
    box3d_x, box3d_y, box3d_z = np.array(box3d_x), np.array(box3d_y), np.array(box3d_z)
    if norm_mode == 'axis_norm':
        box2d_min = np.array([box2d_x[:,0].min(), box2d_y[:,0].min()])[None,None,:]
        box2d_max = np.array([box2d_x[:,1].max(), box2d_y[:,1].max()])[None,None,:]
        box3d_min = np.array([box3d_x[:,0].min(), box3d_y[:,0].min(), box3d_z[:,0].min()])[None,None,:]
        box3d_max = np.array([box3d_x[:,1].max(), box3d_y[:,1].max(), box3d_z[:,1].max()])[None,None,:]
    elif norm_mode == 'all_norm':
        box2d_min = np.array([box2d_x[:,0], box2d_y[:,0]]).min()
        box2d_max = np.array([box2d_x[:,1], box2d_y[:,1]]).max()
        box3d_min = np.array([box3d_x[:,0], box3d_y[:,0], box3d_z[:,0]]).min()
        box3d_max = np.array([box3d_x[:,1], box3d_y[:,1], box3d_z[:,1]]).max()
    else:
        raise ValueError('Invalid `norm_mode` {}'.format(norm_mode))
    for i in range(n_data):
        data['pose']['2d'][i] = (data['pose']['2d'][i] - box2d_min) / (box2d_max - box2d_min)
        data['pose']['3d'][i] = (data['pose']['3d'][i] - box3d_min) / (box3d_max - box3d_min)


def align_and_normalize_dataset_v2(data):
    """ Normalize the entire dataset. [Ver.2] """
    n_seq = len(data['pose']['2d'])
    fix_ground_ceil = False
    debug = False

    # Get all data statistic
    pose_2d_list, pose_3d_list = data['pose']['2d'], data['pose']['3d']
    total_len = reduce(lambda a, b: a + b.shape[0]*b.shape[1], [0]+pose_2d_list)
    all_xy_sum = reduce(lambda a, b: a + np.sum(b, axis=(0, 1)), [np.zeros((2,))]+pose_2d_list)
    all_xyz_sum = reduce(lambda a, b: a + np.sum(b, axis=(0, 1)), [np.zeros((3,))]+pose_3d_list)
    all_xy_mean = all_xy_sum / total_len
    all_xyz_mean = all_xyz_sum / total_len
    all_xy_sqdev = reduce(lambda a, b, c=all_xy_mean[None,None,:]: a + np.sum((b-c)**2, axis=(0, 1)), [np.zeros((2,))]+pose_2d_list) # Sigma_{(x-mean)^2}
    all_xyz_sqdev = reduce(lambda a, b, c=all_xyz_mean[None,None,:]: a + np.sum((b-c)**2, axis=(0, 1)), [np.zeros((3,))]+pose_3d_list) # Sigma_{(x-mean)^2}
    all_xy_std = np.sqrt(all_xy_sqdev / total_len)
    all_xyz_std = np.sqrt(all_xyz_sqdev / total_len)
    if fix_ground_ceil:
        all_y_limit = reduce(lambda a, b: [min(a[0], b[:,:,1].min(axis=(0,1))), max(a[1], b[:,:,1].max(axis=(0,1)))], [[1E8,-1E8]]+pose_3d_list)

    # Standardize all data; convert all data to a (0, scale) normal distribution
    scale = STD_SCALE
    for seq_i in range(n_seq):
        data['pose']['2d'][seq_i] = scale * (data['pose']['2d'][seq_i] - all_xy_mean) / all_xy_std
        if fix_ground_ceil:
            data['pose']['3d'][seq_i][:,:,0] = scale * (data['pose']['3d'][seq_i][:,:,0] - all_xyz_mean[0]) / all_xyz_std[0]
            data['pose']['3d'][seq_i][:,:,1] = ((data['pose']['3d'][seq_i][:,:,1] - all_y_limit[0]) / (all_y_limit[1] - all_y_limit[0]) - 0.5) * scale
            data['pose']['3d'][seq_i][:,:,2] = scale * (data['pose']['3d'][seq_i][:,:,2] - all_xyz_mean[2]) / all_xyz_std[2]
        else:
            data['pose']['3d'][seq_i] = scale * (data['pose']['3d'][seq_i] - all_xyz_mean) / all_xyz_std

    if debug:
        new_xy_mean = reduce(lambda a, b: a + np.sum(b, axis=(0, 1)), [np.zeros((2,))]+data['pose']['2d']) / total_len
        new_xyz_mean = reduce(lambda a, b: a + np.sum(b, axis=(0, 1)), [np.zeros((3,))]+data['pose']['3d']) / total_len
        new_xy_sqdev = reduce(lambda a, b, c=new_xy_mean[None,None,:]: a + np.sum((b-c)**2, axis=(0, 1)), [np.zeros((2,))]+data['pose']['2d']) # Sigma_{(x-mean)^2}
        new_xyz_sqdev = reduce(lambda a, b, c=new_xyz_mean[None,None,:]: a + np.sum((b-c)**2, axis=(0, 1)), [np.zeros((3,))]+data['pose']['3d']) # Sigma_{(x-mean)^2}
        new_xy_std = np.sqrt(new_xy_sqdev / total_len)
        new_xyz_std = np.sqrt(new_xyz_sqdev / total_len)


def get_limb_length(pose, parents):
    """ Compute length of all limbs in 3D poses """
    limb_len = []
    for p_i, p in enumerate(pose):
        if p_i != 0:
            x_len = abs(p[0] - pose[parents[p_i], 0])
            y_len = abs(p[2] - pose[parents[p_i], 2])
            z_len = abs(p[1] - pose[parents[p_i], 1])
            limb_len.append(np.sqrt(x_len**2 + y_len**2 + z_len**2))
    return limb_len


class SkeletonEvaluator(object):
    def __init__(self):
        pass


class Skeleton3DVisualizer(object):
    def __init__(self, parents, plot_3d_limit=[0.0, 1.0], show_joint=False, show_ticks=False, render=False):
        self.parents = parents
        self.plot_3d_limit = plot_3d_limit
        self.camera_azimuth = [70, 70, 110, 110]
        self.show_joint = show_joint
        self.show_ticks = show_ticks
        self.render = render
        self.fig = plt.figure(figsize=(2, 2), dpi=64)
        self.fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
        self.axes = []
        self.dot_3d, self.line_3d = [], []

        # Init 3D plot
        self.axes.append(self.fig.add_subplot(1, 1, 1, projection='3d'))
        ax = self.axes[0]
        if not self.show_ticks:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        else:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        if self.plot_3d_limit is not None:
            ax.set_xlim3d(*plot_3d_limit[::-1])
            ax.set_ylim3d(*plot_3d_limit)
            ax.set_zlim3d(*plot_3d_limit[::-1])
        if self.show_joint:
            for d_i in range(len(self.parents)):
                self.dot_3d.append(ax.plot([0.5], [0.5], [0.5], 'o', zdir='z', c='r', markersize=5))
        for l_i in range(len(self.parents) - 1):
            if l_i in [0, 1, 2, 13, 14, 15]: # right joints. NOTE: for 17-joint skeleton
            #if l_i in [1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]: # right joints. NOTE: for 32-joint skeleton
                color = 'r'
            elif l_i in [3, 4, 5, 10, 11, 12]: # left joints. NOTE: for 17-joint skeleton
            #elif l_i in [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23]: # left joints. NOTE: for 32-joint skeleton
                color = 'b'
            else:
                color = 'g'
            self.line_3d.append(ax.plot([0., 1.], [0., 1.], [0., 1.], zdir='z', c=color, linewidth=3))
        #self.fig.patch.set_facecolor('black')

        if self.render:
            plt.show(block=False)
    
    def set_data(self, pose_3d, camera_view):
        # Reset 3D plot
        ax = self.axes[0]
        data = pose_3d
        ax.view_init(elev=15., azim=self.camera_azimuth[camera_view])
        if self.plot_3d_limit is None:
            ax.set_xlim3d(data[:,:,0].max(), data[:,:,0].min()) # NOTE: reverse axis to fit data
            ax.set_ylim3d(data[:,:,2].min(), data[:,:,2].max())
            ax.set_zlim3d(data[:,:,1].max(), data[:,:,1].min()) # NOTE: reverse axis to fit data

        # Update plots
        imgs = []
        for seq_i, seq_3d in enumerate(pose_3d): # Iterate through sequence
            # Draw the specified joint
            if self.show_joint:
                raise NotImplementedError
                self.dot_3d[0].set_data(seq_3d[self.show_joint, 0], seq_3d[self.show_joint, 2])
                self.dot_3d[0].set_3d_properties(seq_3d[self.show_joint, 1], zdir='z')
            for d_i, d_3d in enumerate(seq_3d): # Iterate through skeleton
                if d_i != 0: # Skip parent[0] --> -1
                    # Draw 3D skeleton
                    self.line_3d[d_i-1][0].set_xdata([d_3d[0], seq_3d[self.parents[d_i], 0]])
                    self.line_3d[d_i-1][0].set_ydata([d_3d[2], seq_3d[self.parents[d_i], 2]])
                    self.line_3d[d_i-1][0].set_3d_properties([d_3d[1], seq_3d[self.parents[d_i], 1]], zdir='z')
            imgs.append(fig2img(self.fig))

            if self.render:
                plt.pause(0.1)
        imgs = np.array(imgs)

        return imgs


def fig2img(fig):
    """ Convert Matplotlib plot to image. """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    # put the figure pixmap into a numpy array
    w, h, d = buf.shape
    img = np.array(Image.frombytes('RGBA', (w, h), buf.tostring()))[:,:,:3]
    crop = 15
    img = img[crop:h-crop,crop:w-crop]
    #img[img == 255] = 0 # NOTE: hacky! manually set background to black
    return img
