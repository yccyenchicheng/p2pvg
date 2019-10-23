import os
import sys
import time
import random
import logging
import argparse
from datetime import datetime

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import itertools
import progressbar
from progressbar import Percentage, Bar, ETA
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from data import data_utils
from misc import utils
from misc import metrics
from misc import visualize
from misc import criterion

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='gpu to use')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--log_dir', default='logs/p2pvg', help='base directory to save logs')
parser.add_argument('--data_root', default='data_root', help='root directory for data')
parser.add_argument('--ckpt', type=str, default='', help='load ckpt for continued training') # load ckpt

parser.add_argument('--dataset', default='mnist', help='dataset to train with (mnist | weizmann | h36m | bair)')
parser.add_argument('--num_digits', type=int, default=1, help='number of digits for moving mnist')
parser.add_argument('--nepochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--epoch_size', type=int, default=300, help='how many batches for 1 epoch')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=22, type=int, help='batch size')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')

parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--n_past', type=int, default=1, help='number of frames to condition on') # NOTE
parser.add_argument('--nsample', type=int, default=20, help='number of samples to generate per test sequence')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t. kth: 32')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--backbone', default='dcgan', help='model type (dcgan | vgg | mlp), mlp for h36m')
parser.add_argument('--last_frame_skip', action='store_true',
                    help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
parser.add_argument('--max_seq_len', type=int, default=30, help='number of dynamic length of frames for training.')
parser.add_argument('--delta_len', type=int, default=5, help='train seq: [max_seq_len-delta_len*2, max_seq_len].')
parser.add_argument('--weight_cpc', type=float, default=1000.0, help='weighting for the L2 loss between cp and our generated frame.')
parser.add_argument('--weight_align', type=float, default=0.0, help='weighting for alignment between latent space from encoder and frame predictor.')
parser.add_argument('--skip_prob', type=float, default=0.1, help='probability to skip a frame in training.')
parser.add_argument('--qual_iter', type=int, default=1, help='frequency to eval the quantitative results.')
parser.add_argument('--quan_iter', type=int, default=1, help='frequency to eval the quantitative results.')
parser.add_argument('--test', action='store_true') # test my code

opt = parser.parse_args()

def train(model, x, start_ix, cp_ix, gen=False):
    model.zero_grad()
    mse_loss, kld_loss, cpc, align_loss = model(x, start_ix, cp_ix)
    return mse_loss, kld_loss, cpc, align_loss

# gpu to use
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.gpu)

# setup log_dir
if opt.ckpt == '':
    log_suffix = {
        'dataset': opt.dataset,
        'cpc': opt.weight_cpc,
        'align': opt.weight_align,
        'skip_prob': opt.skip_prob,
        'batch_size':opt.batch_size,
        'backbone': opt.backbone,
        'beta': opt.beta,
        'g_dim': opt.g_dim,
        'z_dim': opt.z_dim,
        'rnn_size': opt.rnn_size,
        }

    log_name = 'P2PModel'
    for key, val in log_suffix.items():
        log_name += '-{}_{}'.format(key, val)

    opt.log_dir = '%s-%s' % (opt.log_dir, log_name)
    if opt.test:
        opt.log_dir = 'logs/test-%s-%s' % (os.path.basename(opt.log_dir), datetime.now().strftime('%Y-%m-%d_%H-%M'))
else:
    states = torch.load(opt.ckpt)
    opt.log_dir = states['opt'].log_dir

os.makedirs('%s/gen_vis/' % opt.log_dir, exist_ok=True)

# tensorboard writer
tboard_dir = os.path.join(opt.log_dir, 'tboard')
try:
    writer = SummaryWriter(log_dir=tboard_dir)
except:
    writer = SummaryWriter(logdir=tboard_dir)

# setups starts here
# logger
logger = utils.get_logger(logpath=os.path.join(opt.log_dir, 'logs'), filepath=__file__)
logger.info(opt)

# store cmd
cmd = utils.store_cmd(opt=opt)

# set seed
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
logger.info("[*] Random Seed: %d" % opt.seed)
# setups ends here 


# setup datasets 
train_data, test_data = data_utils.load_dataset(opt)
train_generator = data_utils.get_data_generator(train_data, train=True, opt=opt)
test_dl_generator = data_utils.get_data_generator(test_data, train=False, dynamic_length=True, opt=opt)

if opt.dataset == 'h36m':
    from human36m import Skeleton3DVisualizer, STD_SCALE
    visualizer = Skeleton3DVisualizer(train_data.skeleton.parents(), plot_3d_limit=[-2*STD_SCALE,2*STD_SCALE], 
                                      show_joint=False, show_ticks=False, render=False)
else:
    visualizer = None


# set up the models 
if opt.dataset != 'h36m':
    if opt.backbone == 'dcgan':
        if opt.image_width == 64:
            import models.dcgan_64 as backbone_net
        elif opt.image_width == 128:
            import models.dcgan_128 as backbone_net
    elif opt.backbone == 'vgg':
        if opt.image_width == 64:
            import models.vgg_64 as backbone_net
        elif opt.image_width == 128:
            import models.vgg_128 as backbone_net
elif opt.dataset == 'h36m':
    import models.h36m_mlp as backbone_net
else:
    raise ValueError('Unknown backbone: %s' % opt.backbone)
opt.backbone_net = backbone_net

from models.p2p_model import P2PModel

# model
model = P2PModel(opt.batch_size, opt.channels, opt.g_dim, opt.z_dim, opt.rnn_size,
                  opt.prior_rnn_layers, opt.posterior_rnn_layers, opt.predictor_rnn_layers, opt=opt)

# criterions
mse_criterion = nn.MSELoss()
kl_criterion = criterion.KLCriterion()

model.cuda()
mse_criterion.cuda()
kl_criterion.cuda()

if opt.ckpt != '':
    states = torch.load(opt.ckpt)
    start_epoch = model.load(states=states)
    logger.info("[*] Load model from %s. Training continued at: %d" % (opt.ckpt, start_epoch))
else:
    start_epoch = 0

# training

# num of lengths to gen for qualitative results
#qual_lengths = [10, opt.max_seq_len]
qual_lengths = [10, 30]

logger.info('[*] Using gpu: %d' % opt.gpu)
logger.info('[*] log dir: %s' % opt.log_dir)

epoch_size = opt.epoch_size
for epoch in range(start_epoch, opt.nepochs):
    model.train()

    epoch_mse = 0
    epoch_kld = 0
    epoch_align = 0
    epoch_cpc = 0

    progress = utils.get_progress_bar('Training epoch: %d' % epoch, epoch_size)
    for i in range(epoch_size):
        progress.update(i+1)

        x = next(train_generator)

        # train p2p model
        start_ix = 0
        cp_ix = -1
        cp_ix = len(x)-1
        mse, kld, cpc, align = train(model, x, start_ix, cp_ix)
        epoch_mse += mse
        epoch_kld += kld
        epoch_cpc += cpc
        epoch_align += align

        # log training info
        if i % 50 == 0 and i != 0:
            step = epoch * epoch_size + i
            writer.add_scalar("Train/mse", epoch_mse/i, step)
            writer.add_scalar("Train/kld", epoch_kld/i, step)
            writer.add_scalar("Train/cpc", epoch_cpc/i, step)
            writer.add_scalar("Train/align", epoch_align/i, step)

            for name, param in model.named_parameters():
                if param.requires_grad:
                    name = name.replace('.', '/')
                    writer.add_histogram(name, param.data.cpu().numpy(), step)
                    try:
                        writer.add_histogram(name+'/grad', param.grad.data.cpu().numpy(), step)
                    except:
                        pass

    progress.finish()
    utils.clear_progressbar()
    logger.info('[%02d] mse loss: %.5f | kld loss: %.5f | align loss: %.5f | cpc loss: %.5f (%d)' % 
                                                                (epoch, epoch_mse/epoch_size, 
                                                                        epoch_kld/epoch_size,
                                                                        epoch_align/epoch_size,
                                                                        epoch_cpc/epoch_size,
                                                                        epoch*epoch_size*opt.batch_size))
                                                                        
    ###### qualitative results ######
    model.eval()
    with torch.no_grad():
        if (epoch + 1) % opt.qual_iter == 0: # NOTE for fast training if set opt.quan_iter larger
            end = time.time()
            x = next(test_dl_generator)

            if opt.dataset == 'h36m':
                length_to_gen = x[1].shape[0]
            else:
                length_to_gen = len(x)
            visualize.vis_seq(model, x, epoch, length_to_gen, model_mode='full', recon_mode='test', skip_frame=False,
                                h36m_visualizer=visualizer, writer=writer, opt=opt)
            visualize.vis_seq(model, x, epoch, length_to_gen, model_mode='posterior', recon_mode='test', skip_frame=False,
                                h36m_visualizer=visualizer, writer=writer, opt=opt)
            visualize.vis_seq(model, x, epoch, length_to_gen, model_mode='prior', recon_mode='test', skip_frame=False,
                                h36m_visualizer=visualizer, writer=writer, opt=opt)


            for ix, length_to_gen in enumerate(qual_lengths):
                # NOTE do not skip frame for qualitative results
                visualize.vis_seq(model, x, epoch, length_to_gen, model_mode='full', skip_frame=False,
                                  h36m_visualizer=visualizer, writer=writer, opt=opt)
                visualize.vis_seq(model, x, epoch, length_to_gen, model_mode='posterior', skip_frame=False,
                                  h36m_visualizer=visualizer, writer=writer, opt=opt)
                visualize.vis_seq(model, x, epoch, length_to_gen, model_mode='prior', skip_frame=False,
                                  h36m_visualizer=visualizer, writer=writer, opt=opt)

            print("[*] Time for qualitative results: %.4f" % (time.time() - end))
    ###### qualitative results ######

    # save the model
    fname = '%s/model_%d.pth' % (opt.log_dir, epoch)
    model.save(fname, epoch)
    logger.info("[*] Model saved at: %s" % fname)
    os.system("cp %s/model_%d.pth %s/model.pth" % (opt.log_dir, epoch, opt.log_dir)) # latest ckpt

    if epoch % 10 == 0:
        logger.info('log dir: %s' % opt.log_dir)
