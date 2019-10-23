import os
import sys
import math
import socket
import logging
import argparse
import numpy as np
import scipy.misc
import functools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from skimage.measure import compare_psnr as psnr_metric
from skimage.measure import compare_ssim as ssim_metric
from scipy import signal
from scipy import ndimage
from PIL import Image, ImageDraw
import imageio
import torch
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import datasets, transforms

from progressbar import Counter, Percentage, Bar, ETA, ProgressBar, SimpleProgress

def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]

def normalize_data(opt, dtype, sequence):
    if opt.dataset == 'smmnist' or opt.dataset == 'kth' or opt.dataset == 'bair' or opt.dataset == 'dlsmmnist':
        sequence.transpose_(0, 1)
        sequence.transpose_(3, 4).transpose_(2, 3)
    else:
        sequence.transpose_(0, 1)

    return sequence_input(sequence, dtype)

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))

def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result

def save_np_img(fname, x):
    if x.shape[0] == 1:
        x = np.tile(x, (3, 1, 1))
    img = scipy.misc.toimage(x,
                             high=255*x.max(),
                             channel_axis=0)
    img.save(fname)

def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    return scipy.misc.toimage(tensor.numpy(),
                              high=255*tensor.max().numpy(),
                              channel_axis=0)

def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0,0,0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)

def save_gif(filename, inputs, duration=0.25):

    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1)
        img = (img.numpy() * 255.).astype(np.uint8)
        images.append(img)
    imageio.mimsave(filename, images, duration=duration)

def save_gif_with_text(filename, inputs, text, duration=0.25):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1).numpy()
        images.append(img)
    imageio.mimsave(filename, images, duration=duration)

def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)

def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)

def prod(l):
    return functools.reduce(lambda x, y: x * y, l)

def batch_flatten(x):
    return x.resize(x.size(0), prod(x.size()[1:]))

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# ------ added by yenchi
def plt_fig_to_tensor(inputs):
    fig = plt.figure()
    canvas = FigureCanvas(fig)
    plt.plot([i for i in range(len(inputs))], inputs)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    inputs_np = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
    inputs_tensor = torch.from_numpy(inputs_np).permute(2, 0, 1).float() / 255.
    plt.close()
    return inputs_tensor

def gifs_to_tensor(gifs):
    """Turn a list of list of list of tensor to a tensor.

    params:
        gifs: gifs[t][row_block][col] is a image with (c, h, w)

    returns:
        movie_tensor: a tensor where
        1. First the image concatenated along the length: gifs[t][row_block]: (c, h, w*col)
        2. Then concate along the h and split by *batch_size*: gifs[t]: (c)
    
    """
    stacked_gifs = stack_inner(gifs).detach()
    T, b, cols, c, h, w = stacked_gifs.shape
    # cat w
    stacked_gifs = stacked_gifs.view(-1, cols, c, h, w) # --> T*b, cols, c, h, w
    stacked_gifs = stacked_gifs.permute(0, 1, 4, 2, 3).contiguous() # --> T*b, cols, w, c, h
    w_cat = stacked_gifs.view(-1, cols*w, c, h).permute(0, 2, 3, 1) # --> T*b, cols*w, c, h --> T*b, c, h, cols*w
    # cat h
    w_cat = w_cat.view(T, b, c, h, cols*w)
    w_cat = w_cat.permute(0, 1, 3, 2, 4).contiguous() # --> T, b, h, c, cols*w
    h_cat = w_cat.view(-1, b*h, c, cols*w).permute(0, 2, 1, 3).contiguous() # --> T, c, b*h, cols*w

    video_tensor = h_cat[None, ...]

    return video_tensor

def stack_inner(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs
    else:
        return torch.stack([stack_inner(x) for x in inputs])

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def store_cmd(opt=None):
    argv = sys.argv
    cmd = "python"

    for i, arg in enumerate(argv):
        cmd += " %s" % arg
        if arg == "--log_dir":
            model_dir = argv[i+1]

    cmd_continued = cmd + " --model_dir %s" % opt.log_dir

    with open("%s/cmd.txt" % opt.log_dir, "w") as f:
        f.write("%s\n" % cmd)
        f.write("%s\n" % cmd_continued)
    return cmd

def get_progress_bar(msg, max_val):
    return ProgressBar(widgets=['[*] %s' % msg, Percentage(), Bar(), ETA()], max_value=max_val).start()

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

def make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)