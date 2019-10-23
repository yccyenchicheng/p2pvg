import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DynamicLengthMovingMNIST(object):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""
    def __init__(self, data_root='data_root', train=True, transform=None, max_seq_len=20, n_past=1, 
                 delta_len=3, image_size=64, num_digits=2, deterministic=True, opt=None):
        self.data_root = data_root
        self.train = train
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.n_past = n_past
        self.delta_len = delta_len
        self.image_size = image_size 

        self.num_digits = num_digits  
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1
        self.digit_size = 32
        self.opt = opt

        if self.transform is None:
            self.transform = transforms.Compose(
                                [transforms.Scale(self.digit_size),
                                 transforms.ToTensor()])

        self.data = datasets.MNIST(
            self.data_root,
            train=self.train,
            download=True,
            transform=self.transform)

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def get_seq_len(self):
        seq_len = np.random.randint(low=self.max_seq_len-self.delta_len*2, high=self.max_seq_len+1)
        return seq_len

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size

        x = torch.zeros(self.max_seq_len,
                        self.channels,
                        image_size,
                        image_size)
                       
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.max_seq_len): # I am so dumb...
                if sy < 0:
                    sy = 0 
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size-32:
                    sy = image_size-32-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)
                    
                if sx < 0:
                    sx = 0 
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size-32:
                    sx = image_size-32-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)
                   
                x[t, 0, sy:sy+32, sx:sx+32] += digit[0]
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x
