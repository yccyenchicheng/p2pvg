import imageio
import numpy as np

import torch
import torchvision.utils as vutils

from misc import utils

def save_utils(seq, pad=2, pad_val=0.1, name='test.png'):
    ss = vutils.make_grid(seq, nrow=len(seq), padding=pad, pad_value=pad_val)
    vutils.save_image(ss, name)

def add_gt_cp_border(seq, seq_len, output_len, padding=3):
    """
    Will add orange for start frame, red for end frame.

    params:
        seq: tensor with shape (t, b, c, h, w)
        padding: pixels to pad
    """
    assert len(seq.shape) == 5

    t, b, c, h, w = seq.shape
    if c == 1:
        seq = seq.repeat(1, 1, 3, 1, 1)

    start_ix = 0
    end_ix = seq_len-1

    x_start = seq[start_ix]
    x_end = seq[end_ix]

    # make orange border
    x_start_border = torch.zeros_like(x_start)
    x_start_border[:, 0] = 1.
    x_start_border[:, 1] = 165./255.

    x_start_border[:, :, padding:w-padding, padding:w-padding] = x_start[:, :, padding:w-padding, padding:w-padding]
    
    # make red border
    x_end_border = torch.zeros_like(x_end)
    x_end_border[:, 0] = 1.
    x_end_border[:, :, padding:w-padding, padding:w-padding] = x_end[:, :, padding:w-padding, padding:w-padding]

    seq[start_ix] = x_start_border
    seq[end_ix] = x_end_border

    # already pad the gt_seq so that seq_len = output_len. hence also replace the padded frame here
    if output_len > seq_len:
        for i in range(seq_len, output_len):
            seq[i] = x_end_border

    return seq

def add_samples_cp_border(seq_samples, seq_len, output_len, padding=3):
    """Will add orange for start frame, red for end frame.
    params:
        seq_samples: samples with shape (nsample, t, b, c, h, w)
        padding: pixels to pad
    
    """
    assert len(seq_samples.shape) == 6

    ns, t, b, c, h, w = seq_samples.shape
    if c == 1:
        seq_samples = seq_samples.repeat(1, 1, 1, 3, 1, 1)
    start_ix = 0
    end_ix = output_len-1

    x_start = seq_samples[:, start_ix]
    x_end = seq_samples[:, end_ix]

    # make red border
    x_start_border = torch.zeros_like(x_start)
    x_start_border[:, :, 0] = 1.
    x_start_border[:, :, 1] = 165./255.
    x_start_border[:, :, :, padding:w-padding, padding:w-padding] = x_start[:, :, :, padding:w-padding, padding:w-padding]
    
    # make orange border
    x_end_border = torch.zeros_like(x_end)
    x_end_border[:, :, 0] = 1.
    x_end_border[:, :, :, padding:w-padding, padding:w-padding] = x_end[:, :, :, padding:w-padding, padding:w-padding]

    seq_samples[:, start_ix] = x_start_border
    seq_samples[:, end_ix] = x_end_border

    return seq_samples


def vis_seq(model, x, epoch, output_len, model_mode='full', recon_mode=None, skip_frame=True, h36m_visualizer=None, writer=None, opt=None):
    """ generate a seq with given lengths and input.

    params:
        x: image sequence with shape: (t, b, c, h, w)
        output_len: sequence length of the output, including the start- and end-frame
        model_mode: use 'full model', 'prior' or 'posterior' to predict.
        recon_mode: generate the same length as the input sequence. 
            - 'train': training sequence
            - 'test' as testing sequence.
            - None: generate based on `output_len`
    
    """
    nsample = opt.nsample
    grid_padding = 0   # value passed to make_grid
    nrow_per_block = 6 # nrow to draw per row block (include gt)

    gen_samples = [] # gen_samples[s][t][b_ix]: s for nsample_ix, t for time_ix, b_ix for batch_ix

    start_ix = 0
    if opt.dataset == 'h36m':
        n_block = min(opt.batch_size, 5) # generates at most 10 row block 
        (pose_2d, pose_3d, camera_view) = x
        x = list(x)
        pose_2d, pose_3d, camera_view = pose_2d[:,:n_block], pose_3d[:,:n_block], camera_view[:n_block]
        x = tuple([pose_2d, pose_3d, camera_view])
        gt_seq = [pose_3d[i] for i in range(len(pose_3d))]
        seq_len = len(pose_3d)
        cp_ix = seq_len-1
        x_cp = pose_3d[cp_ix]
    else:
        n_block = min(opt.batch_size, 10) # generates at most 10 row block 
        gt_seq = [x[i] for i in range(len(x))]
        seq_len = len(x)
        cp_ix = seq_len-1
        x_cp = x[cp_ix]

    eval_cp_ix = output_len-1

    # append additional images so that it has length of eval_seq_len
    for i in range(seq_len, output_len):
        gt_seq.append(x_cp)

    # generate *nsample* sequences
    pbar = utils.get_progress_bar('[%d, %d] %s Qual.' % (epoch, output_len, model_mode), nsample)
    for s in range(nsample):
        # gt_seq_eval, gen_seq_eval: gt/generated frames after n_past
        gen_seq = model.p2p_generate(x, output_len, eval_cp_ix, start_ix=start_ix, cp_ix=cp_ix, model_mode=model_mode, skip_frame=skip_frame)
        if isinstance(gen_seq, list):
            gen_seq = torch.stack(gen_seq) # as tensor with shape: t, b, c, h, w

        #gen_samples[s] = gen_seq
        gen_samples.append(gen_seq)
        pbar.update(s+1)
    utils.clear_progressbar()

    if opt.dataset == 'h36m':
        # Draw sampled sequences
        pbar = utils.get_progress_bar('Draw:', nsample)
        for s in range(nsample):
            #gen_seq_inb = torch.stack(gen_samples[s])
            gen_seq_inb = gen_samples[s]
            imgs_inb = []

            for b in range(gen_seq_inb.shape[1]):
                gen_seq = gen_seq_inb[:, b]
                imgs_inb.append(h36m_visualizer.set_data(gen_seq.data.cpu().numpy(), camera_view[b].item()))
            imgs_inb = list(zip(*imgs_inb)) # swap sequence and batch dimension
            for x_i, x in enumerate(imgs_inb):
                imgs_inb[x_i] = torch.Tensor(np.stack(x).astype(np.float) / 255.).permute(0,3,1,2)
            #gen_samples[s] = imgs_inb # NOTE
            gen_samples[s] = torch.stack(imgs_inb)
            pbar.update(s+1)
        pbar.finish()
        utils.clear_progressbar()

        # Draw ground truth
        gt_seq = torch.stack(gt_seq)
        imgs_inb = []
        for b in range(gt_seq.shape[1]):
            imgs_inb.append(h36m_visualizer.set_data(gt_seq[:,b].data.cpu().numpy(), camera_view[b].item()))
        imgs_inb = list(zip(*imgs_inb)) # swap sequence and batch dimension
        for x_i, x in enumerate(imgs_inb):
            imgs_inb[x_i] = torch.Tensor(np.stack(x).astype(np.float) / 255.).permute(0,3,1,2)
        gt_seq = imgs_inb

    gen_samples = torch.stack(gen_samples)

    # start drawing
    r_len = max(seq_len, output_len)
    gt_seq = torch.stack(gt_seq)

    # highlight the x_cp
    gt_seq = add_gt_cp_border(gt_seq, seq_len=seq_len, output_len=output_len)
    gen_samples = add_samples_cp_border(gen_samples, seq_len=seq_len, output_len=output_len)

    # draw image
    img_canvas = []
    all_row_block = []
    for i in range(n_block):
        row_block = []

        # draw gt img
        #r_gt = vutils.make_grid(x[:, i], nrow=r_len)
        gt = gt_seq[:, i]
        row_block.append(gt)

        # draw gen img
        min_idx = 1
        s_list = [min_idx] + list(np.random.randint(nsample, size=nrow_per_block-1-1)) # size: minus gt and min_cp

        drawn_samples = gen_samples[s_list]

        for j in range(len(s_list)):
            sample_j = drawn_samples[j]

            # pad generated end_frame for all samples if necessary
            if r_len > len(drawn_samples[0]):
                sample_cp = sample_j[eval_cp_ix]
                pad_img = sample_cp.repeat(r_len-len(sample_j), 1, 1, 1, 1)
                sample_j = torch.cat([sample_j, pad_img])
            
            #rj = vutils.make_grid(sample_j[:, i], nrow=r_len)
            s = sample_j[:, i]
            row_block.append(s)

        row_block = torch.stack(row_block) # shape: nrow_per_block, t, c, h, w
        row_block_img = []

        for j in range(nrow_per_block):
            rj = vutils.make_grid(row_block[j], nrow=r_len, padding=grid_padding)
            row_block_img.append(rj)

        row_block_img = torch.cat(row_block_img, dim=1)
        img_canvas.append(row_block_img)

        # NOTE: collect all row_block for gif
        all_row_block.append(row_block)

    # write to disk
    img = torch.cat(img_canvas, dim=1)
    if recon_mode == 'train' or recon_mode == 'test':
        fname = '%s/gen_vis/recon_%s-model_%s-len_%d-epoch_%d.png' % (opt.log_dir, recon_mode, model_mode, output_len, epoch)
    else:
        fname = '%s/gen_vis/gen-model_%s-len-%d-epoch_%d.png' % (opt.log_dir, model_mode, output_len, epoch)
    vutils.save_image(img, fname)

    # draw gif
    vid_tensor = [] # for tensorboard
    gif_canvas = [] # for .gif
    all_row_block = torch.stack(all_row_block) # n_block, nrow_per_block, t, c, h, w

    for t in range(r_len):
        cols = []
        for c in range(nrow_per_block):
            col_i = vutils.make_grid(all_row_block[:, c, t], nrow=1, padding=grid_padding)
            cols.append(col_i)

        frame_t = torch.cat(cols, dim=2)
        vid_tensor.append(frame_t)

        frame_t_np = (frame_t.permute(1, 2, 0).data.cpu().numpy() * 255).astype(np.uint8)
        gif_canvas.append(frame_t_np)

    vid_tensor = torch.stack(vid_tensor).unsqueeze(dim=0) # require shape: N, T, c, h, w

    # write to disk
    if recon_mode == 'train' or recon_mode == 'test':
        fname = '%s/gen_vis/recon_%s-model_%s-len_%d-epoch_%d.gif' % (opt.log_dir, recon_mode, model_mode, output_len, epoch)
    else:
        fname = '%s/gen_vis/gen-model_%s-len-%d-epoch_%d.gif' % (opt.log_dir, model_mode, output_len, epoch)
    imageio.mimsave(fname, gif_canvas)

    # log on tensorboard
    if recon_mode == 'train' or recon_mode == 'test':
        img_tag = "%s/%s-Gen" % (model_mode, recon_mode)
        vid_tag = "%s/%s-Video" % (model_mode, recon_mode)
    else:
        img_tag = "%s/Gen%d" % (model_mode, output_len)
        vid_tag = "%s/GenVideo%d" % (model_mode, output_len)

    writer.add_image(img_tag, img.cpu().numpy(), epoch)
    writer.add_video(vid_tag, vid_tensor.cpu().numpy(), epoch, fps=2)