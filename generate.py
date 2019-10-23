import os
import sys
import time
import random
import argparse

import imageio
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from PIL import Image

import torch
import torch.nn as nn
import torchvision.utils as vutils

from data import data_utils
from misc import utils
from misc import visualize

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='', help='your model.pth file')
parser.add_argument('--video', type=str, default='', help='your .mp4 video file')
parser.add_argument('--output_root', type=str, default='gen_outputs')
parser.add_argument('--seed', type=int, default=1, help='seed to use')

args = parser.parse_args()

def read_video(vid_name):
    """return a torch tensor with shape=(t, b, c, h, w)"""
    reader = imageio.get_reader(vid_name)
    vid_tensor = []
    for i, im in enumerate(reader):
        im = (im/255.).astype(np.float32)
        vid_tensor.append(torch.from_numpy(im))

    ret = torch.stack(vid_tensor).permute(0, 3, 1, 2)
    ret = torch.unsqueeze(ret, 1)
    return ret

def make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)

if __name__ == '__main__':
    states = torch.load(args.ckpt)
    states_opt = states['opt']

    # ------ set up the models ------
    if states_opt.dataset != 'h36m':
        if states_opt.backbone == 'dcgan':
            if states_opt.image_width == 64:
                import models.dcgan_64 as backbone_net
            elif states_opt.image_width == 128:
                import models.dcgan_128 as backbone_net
        elif states_opt.backbone == 'vgg':
            if states_opt.image_width == 64:
                import models.vgg_64 as backbone_net
            elif states_opt.image_width == 128:
                import models.vgg_128 as backbone_net
    elif states_opt.dataset == 'h36m':
        import models.h36m_mlp as backbone
    else:
        raise ValueError('Unknown backbone: %s' % states_opt.backbone)
    states_opt.backbone_net = backbone_net

    from models.p2p_model import P2PModel

    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # model
    batch_size = 1
    model = P2PModel(batch_size, states_opt.channels, states_opt.g_dim, states_opt.z_dim, 
                      states_opt.rnn_size, states_opt.prior_rnn_layers, states_opt.posterior_rnn_layers, 
                      states_opt.predictor_rnn_layers, opt=states_opt)

    model.cuda()
    model.load(states=states)
    model.eval()

    nsamples = 5
    ndisplays = 5 
    assert ndisplays <= nsamples

    gen_lenths = [10, 20, 30]

    # input
    if args.video != '':
        seq = read_video(args.video)
    elif args.start_img != '':
        assert args.end_img != ''
        start = Image.open(args.start_img)
        end = Image.open(args.end_img)
        seq = torch.stack([start, end])
        seq = torch.unsqueeze(seq, 1) # unsqueeze batch dim
    seq = seq.cuda()

    seq_len = len(seq)

    # output path

    output_root = args.output_root
    if output_root == '':
        output_root = 'gen_outputs'
    make_dirs(output_root)

    for length_to_gen in gen_lenths:
        output_cp_ix = length_to_gen - 1

        samples = []
        # maybe make a block
        for s in range(nsamples):
            out = model.p2p_generate(seq, length_to_gen, output_cp_ix, model_mode='full')
            out = torch.stack(out)
            samples.append(out)
        
        samples = torch.stack(samples)

        idx = np.random.choice(len(samples), ndisplays, replace=False)
        samples_to_save = samples[idx]

        # pad gt if necessary
        padded_seq = seq.clone()
        x_cp = seq[seq_len-1]
        if length_to_gen > seq_len:
            pad_frames = x_cp.repeat(length_to_gen-seq_len, 1, 1, 1, 1)
            padded_seq = torch.cat([padded_seq, pad_frames], dim=0)

        # add cp border
        seq_with_border = visualize.add_gt_cp_border(padded_seq, seq_len, length_to_gen)
        samples_to_save = visualize.add_samples_cp_border(samples_to_save, seq_len, length_to_gen)

        # save as img
        seq_grid = vutils.make_grid(seq_with_border[:, 0], nrow=len(seq_with_border), padding=0)
        name = '%s/len_%d-gt.png' % (output_root, length_to_gen)
        vutils.save_image(seq_grid, name)

        block = []
        for ix, s in enumerate(samples_to_save):
            name = '%s/len_%d-gen_%03d.png' % (output_root, length_to_gen, ix)
            s_row = vutils.make_grid(s[:, 0], nrow=len(s), padding=0)
            vutils.save_image(s_row, name)
            block.append(s_row)
        block = torch.cat(block, 1)
        name = '%s/len_%d-gen_full.png' % (output_root, length_to_gen)
        vutils.save_image(block, name)

        # save as gif or mp4
        for ix, s in enumerate(samples_to_save):
            frames = []
            for t in range(len(s)):
                frame_np = (s[t, 0].permute(1, 2, 0).data.cpu().numpy() * 255).astype(np.uint8)
                frames.append(frame_np)
            name = '%s/len_%d-gen_%03d.gif' % (output_root, length_to_gen, ix)
            imageio.mimsave(name, frames)

        gifs = []
        for t in range(length_to_gen):
            col = vutils.make_grid(samples_to_save[:, t, 0], nrow=ndisplays, padding=0)
            col_np = (col.permute(1, 2, 0).data.cpu().numpy() * 255).astype(np.uint8)
            gifs.append(col_np)
        name = '%s/len_%d-gen_full.gif' % (output_root, length_to_gen)
        imageio.mimsave(name, gifs)