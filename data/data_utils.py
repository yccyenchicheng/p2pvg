import os
import sys

from torch.utils.data import DataLoader

def load_dataset(opt, eval=False, eval_len=None, id_act=None):
    if opt.dataset == 'mnist':
        from data.moving_mnist import DynamicLengthMovingMNIST
        
        train_data = DynamicLengthMovingMNIST(
                train=True,
                data_root=opt.data_root,
                max_seq_len=opt.max_seq_len,
                delta_len=opt.delta_len,
                image_size=opt.image_width,
                deterministic=False,
                num_digits=opt.num_digits)
        test_data = DynamicLengthMovingMNIST(
                train=False,
                data_root=opt.data_root,
                max_seq_len=opt.max_seq_len,
                delta_len=opt.delta_len,
                image_size=opt.image_width,
                deterministic=False,
                num_digits=opt.num_digits)

    elif opt.dataset == 'weizmann':
        assert opt.channels == 3, "=> %s has 3 channels, but opt.channels = %d" % (opt.dataset, opt.channels)
        from data.weizmann import WeizmannDataset
        train_max_seq_len = 18
        test_max_seq_len  = 10

        train_data = WeizmannDataset(
                data_root=opt.data_root,
                train=True,
                max_seq_len=train_max_seq_len,
                n_past=opt.n_past,
                delta_len=opt.delta_len,
                image_size=opt.image_width,
                opt=opt)
        test_data = WeizmannDataset(
                data_root=opt.data_root,
                train=False,
                max_seq_len=test_max_seq_len,
                n_past=opt.n_past,
                delta_len=opt.delta_len,
                image_size=opt.image_width,
                opt=opt)
                    
    elif opt.dataset == 'h36m':
        sys.path.append('../human36m') # original location of train.py: ~/ICCV19-baselines/svg_cp_aware/train_xxx.py
        sys.path.append('data/human36m')
        from human36m import Human36mDataset

        max_seq_len = 30
        speed_range = [6, 6]
        n_breakpoints = 0
        acc_range = [0, 0]
        opt.data_root = os.path.join(opt.data_root, 'processed/h36m-fetch/processed')

        train_data = Human36mDataset(data_root=opt.data_root,
                                    max_seq_len=max_seq_len,
                                    delta_len=opt.delta_len,
                                    speed_range=speed_range,
                                    n_breakpoints=n_breakpoints,
                                    acc_range=acc_range,
                                    mode='train')
        test_data = Human36mDataset(data_root=opt.data_root,
                                    max_seq_len=max_seq_len,
                                    delta_len=opt.delta_len,
                                    speed_range=[1, 1],
                                    n_breakpoints=0,
                                    acc_range=[0, 0],
                                    mode='test')
    elif opt.dataset == 'bair':
        assert opt.channels == 3, "=> %s has 3 channels, but opt.channels = %d" % (opt.dataset, opt.channels)
        from data.bair import BairRobotPush

        train_data = BairRobotPush(
            data_root=opt.data_root,
            train=True,
            max_seq_len=opt.max_seq_len,
            delta_len=opt.delta_len,
            image_size=opt.image_width)
        test_data = BairRobotPush(
            data_root=opt.data_root,
            train=False,
            max_seq_len=opt.max_seq_len,
            delta_len=opt.delta_len,
            image_size=opt.image_width)

    return train_data, test_data

def get_h36m_generator(loader, dynamic_length=True, opt=None):
    while True:
        # should reset_seq_len before getting sequence
        for i, data in enumerate(loader):
            seq_len = loader.dataset.get_seq_len()
            pose_2d = data['pose_2d'].permute(1, 0, 2, 3).float().cuda()
            pose_3d = data['pose_3d'].permute(1, 0, 2, 3).float().cuda()
            camera_view = data['camera_view']
            speed = data['speed']
            breakpoints = data['breakpoints']

            # dynamic length
            if dynamic_length:
                pose_2d = pose_2d[:seq_len]
                pose_3d = pose_3d[:seq_len]
            yield (pose_2d, pose_3d, camera_view)


def get_generator(loader, dynamic_length=True, opt=None):
    while True:
        for i, data in enumerate(loader):
            # time first
            data = data.permute(1, 0, 2, 3, 4).cuda()
            seq_len = loader.dataset.get_seq_len()

            if dynamic_length:
                data = data[:seq_len]
            
            yield data

def get_data_generator(data, train=True, dynamic_length=True, opt=None):

    # dataloader
    if opt.dataset == 'h36m':
        if train:
            loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=1)
        else:
            loader = DataLoader(data, batch_size=10, shuffle=True, drop_last=True, num_workers=1)
        generator = get_h36m_generator(loader, dynamic_length=dynamic_length, opt=opt)
    else:
        if train:
            loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=1)
        else:
            loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=1)
        
        generator = get_generator(loader, dynamic_length=dynamic_length, opt=opt)

    return generator