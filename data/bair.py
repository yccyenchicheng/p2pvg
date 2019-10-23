import os
import io
import numpy as np
from PIL import Image
from scipy.misc import imresize
from scipy.misc import imread 

import torch
import torchvision.transforms as transforms

class BairRobotPush(object):
    """ Our robot pushing dataloader"""
    def __init__(self, data_root, train=True, transform=None, max_seq_len=30, delta_len=5, image_size=64, opt=None):
        self.root_dir = os.path.join(data_root, 'bair')
        self.train = train
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.delta_len = delta_len
        self.image_size = image_size
        self.seed_is_set = False # multi threaded loading

        if train:
            self.data_dir = '%s/processed_data/train' % self.root_dir
            self.ordered = False
        else:
            self.data_dir = '%s/processed_data/test' % self.root_dir
            self.ordered = True 
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))

        self.d = 0
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def get_seq_len(self):
        seq_len = np.random.randint(low=self.max_seq_len-self.delta_len*2, high=self.max_seq_len+1)
        return seq_len

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return 10000

    def get_seq(self):
        if self.ordered:
            d = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d+=1
        else:
            d = self.dirs[np.random.randint(len(self.dirs))]

        image_seq = torch.zeros(self.max_seq_len, 3, 64, 64)

        for i in range(self.max_seq_len):
            fname = '%s/%d.png' % (d, i)
            im = Image.open(fname)
            if self.transform is not None:
                im = self.transform(im)
            
            image_seq[i] = im

        return image_seq

    def __getitem__(self, index):
        self.set_seed(index)
        return self.get_seq()
