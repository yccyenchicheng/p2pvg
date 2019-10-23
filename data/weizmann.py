import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset

class WeizmannDataset(Dataset):
    def __init__(self, data_root='data_root', train=True, transform=None, max_seq_len=18, n_past=1,
                 delta_len=3, image_size=64, opt=None):
        self.root = os.path.join(data_root, 'weizmann')
        self.train = train
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.n_past = n_past
        self.delta_len = delta_len

        self.channels = 3
        self.image_size = image_size
        self.seed_is_set = False
        self.flip = transforms.RandomHorizontalFlip(p=1.0)
        self.opt = opt

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        self.data = []

        images_dir = os.path.join(self.root)
        ids = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]

        # loop through people
        for identity in ids:
            actions = sorted(os.listdir(os.path.join(self.root, identity)))

            # loop though action
            for act in actions:
                frames = sorted(os.listdir(os.path.join(self.root, identity, act)))

                num_train = len(frames) * 2 // 3

                if self.train:
                    # min length: 18
                    start = 0
                    end = num_train
                else:
                    # min length: 10
                    start = num_train
                    end = len(frames)
                
                n_frames = end - start

                if n_frames < max_seq_len:
                    continue

                # T, c, h, w
                seq = torch.zeros(n_frames, self.channels, self.image_size, self.image_size)
                seq_flip = torch.zeros(n_frames, self.channels, self.image_size, self.image_size) # flip a sequence here

                for t in range(start, end):
                    img_name = frames[t]
                    img_path = os.path.join(self.root, identity, act, img_name)
                    im = Image.open(img_path)
                    
                    if self.transform:
                        im_flip = self.flip(im)
                        im = self.transform(im)
                        im_flip = self.transform(im_flip)

                    seq[t-start] = im

                    # flipped seq
                    seq_flip[t-start] = im_flip

                self.data.append({
                    "seq": seq,
                    "n_frames": n_frames,
                })

                self.data.append({
                    "seq": seq_flip,
                    "n_frames": n_frames,
                })

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def get_seq_len(self):
        if self.train:
            seq_len = np.random.randint(low=10, high=self.max_seq_len+1)
        else:
            seq_len = np.random.randint(low=6, high=self.max_seq_len+1)

        return seq_len

    def __getitem__(self, idx):
        self.set_seed(idx)

        data = self.data[idx]
        seq = data["seq"]
        n_frames = data["n_frames"]

        start_ix = np.random.randint(low=0, high=n_frames-self.max_seq_len+1)
        end_ix   = start_ix + self.max_seq_len

        seq = seq[start_ix:end_ix]
        return seq

    def __len__(self):
        return len(self.data)
