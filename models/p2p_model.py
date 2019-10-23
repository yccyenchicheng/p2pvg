import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

import models.lstm as lstm_models
from misc import criterion
from misc import utils

class P2PModel(nn.Module):
    def __init__(self, batch_size=100, channels=1, g_dim=128, z_dim=10, rnn_size=256, 
                 prior_rnn_layers=1, posterior_rnn_layers=1, predictor_rnn_layers=2, opt=None):

        super().__init__()
        self.batch_size           = batch_size
        self.channels             = channels
        self.g_dim                = g_dim
        self.z_dim                = z_dim
        self.rnn_size             = rnn_size
        self.prior_rnn_layers     = prior_rnn_layers
        self.posterior_rnn_layers = posterior_rnn_layers
        self.predictor_rnn_layers = predictor_rnn_layers
        self.opt                  = opt

        # LSTMs
        self.frame_predictor = lstm_models.lstm(self.g_dim+self.z_dim+1+1, self.g_dim, self.rnn_size, self.predictor_rnn_layers, self.batch_size)
        self.posterior = lstm_models.gaussian_lstm(self.g_dim+self.g_dim+1+1, self.z_dim, self.rnn_size, self.posterior_rnn_layers, self.batch_size)
        self.prior = lstm_models.gaussian_lstm(self.g_dim+self.g_dim+1+1, self.z_dim, self.rnn_size, self.prior_rnn_layers, self.batch_size)

        # encoder & decoder
        if opt.dataset == 'h36m':
            self.encoder = opt.backbone_net.encoder(out_dim=self.g_dim, h_dim=self.g_dim)
            self.decoder = opt.backbone_net.decoder(in_dim=self.g_dim, h_dim=self.g_dim)
        else:
            self.encoder = opt.backbone_net.encoder(self.g_dim, self.channels)
            self.decoder = opt.backbone_net.decoder(self.g_dim, self.channels)

        # optimizer
        opt.optimizer = optim.Adam

        # criterions
        self.mse_criterion = nn.MSELoss() # recon and cpc
        self.kl_criterion = criterion.KLCriterion(opt=self.opt)
        self.align_criterion = nn.MSELoss()
        
        self.init_weight()
        self.init_optimizer()

    def init_optimizer(self):
        opt = self.opt
        self.frame_predictor_optimizer = opt.optimizer(self.frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.posterior_optimizer = opt.optimizer(self.posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.prior_optimizer = opt.optimizer(self.prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.encoder_optimizer = opt.optimizer(self.encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.decoder_optimizer = opt.optimizer(self.decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def init_hidden(self, batch_size=1):
        self.frame_predictor.hidden = self.frame_predictor.init_hidden(batch_size=batch_size)
        self.posterior.hidden       = self.posterior.init_hidden(batch_size=batch_size)
        self.prior.hidden           = self.prior.init_hidden(batch_size=batch_size)

    def init_weight(self):
        self.frame_predictor.apply(utils.init_weights)
        self.posterior.apply(utils.init_weights)
        self.prior.apply(utils.init_weights)
        self.encoder.apply(utils.init_weights)
        self.decoder.apply(utils.init_weights)

    def get_global_descriptor(self, x, start_ix=0, cp_ix=None):
        """Get the global descriptor based on x, start_ix, cp_ix."""
        if cp_ix is None:
            cp_ix = len(x) - 1
        x_cp = x[cp_ix]
        h_cp = self.encoder(x_cp)[0] # 1 is input for skip-connection

        return x_cp, h_cp
        
    def p2p_generate(self, x, len_output, eval_cp_ix, start_ix=0, cp_ix=-1, model_mode='full', 
                 skip_frame=False, init_hidden=True):
        """Point-to-Point Generation given input sequence. Generate *1* sample for each input sequence.

        params:
            x: input sequence
            len_output: length of the generated sequence
            eval_cp_ix: cp_ix of the output sequence. usually it is len_output-1
            model_mode:
                - full:      post then prior
                - posterior: all use posterior
                - prior:     all use prior

        """
        opt = self.opt

        if type(x) == tuple:
            # h36m
            (pose_2d, pose_3d, camera_view) = x
            #T, bs = len(pose_3d), pose_3d[0].shape[0]
            #x = pose_3d.view(T, bs, -1)
            x = pose_3d
            batch_size, coor, n_dim = x[0].shape
            dim_shape = (coor, n_dim)
        else:
            batch_size, channels, h, w = x[0].shape
            dim_shape = (channels, h, w)

        # gen_seq will collect the generated frames
        gen_seq = [x[0]]
        x_in = x[0]

        # NOTE: for visualization
        # init lstm
        if init_hidden:
            self.init_hidden(batch_size=batch_size)

        # get global descriptor
        seq_len        = len(x)
        cp_ix          = seq_len-1
        x_cp, global_z = self.get_global_descriptor(x, cp_ix=cp_ix) # here global_z is h_cp

        ###### time skipping
        skip_prob = opt.skip_prob

        prev_i = 0
        max_skip_count = seq_len * skip_prob
        skip_count = 0
        probs = np.random.uniform(0, 1, len_output-1)

        # for each sample, generate *n_eval* frames
        for i in range(1, len_output):
            #if np.random.uniform(0, 1) <= skip_prob and i > 1 and skip_count < max_skip_count and i != cp_ix:
            if (probs[i-1] <= skip_prob and i >= opt.n_past and skip_count < max_skip_count 
                                        and i != 1 and i != (len_output-1) and skip_frame): 
                skip_count += 1
                gen_seq.append(torch.zeros_like(x_in))
                continue

            time_until_cp = torch.zeros(batch_size, 1).fill_((eval_cp_ix-i+1)/eval_cp_ix).to(x_cp)
            delta_time = torch.zeros(batch_size, 1).fill_((i-prev_i)/eval_cp_ix).to(x_cp)

            prev_i = i

            h = self.encoder(x_in)
            #if opt.last_frame_skip or i < opt.n_past: # original
            if opt.last_frame_skip or i == 1 or i < opt.n_past:
                h, skip = h
            else:
                h, _ = h

            h_cpaw = torch.cat([h, global_z, time_until_cp, delta_time], 1).detach()

            if i < opt.n_past:
                h_target = self.encoder(x[i])[0]
                h_target_cpaw = torch.cat([h_target, global_z, time_until_cp, delta_time], 1).detach()
                zt  , _, _ = self.posterior(h_target_cpaw)
                zt_p, _, _ = self.prior(h_cpaw)

                if model_mode == 'posterior' or model_mode == 'full':
                    self.frame_predictor(torch.cat([h, zt, time_until_cp, delta_time], 1))
                elif model_mode == 'prior':
                    self.frame_predictor(torch.cat([h, zt_p, time_until_cp, delta_time], 1))
                
                x_in = x[i]
                gen_seq.append(x_in) # NOTE: gen_seq can append the decoded x_in for comparing with gt
            else:
                if i < len(x): # for posterior
                    h_target = self.encoder(x[i])[0]
                    h_target_cpaw = torch.cat([h_target, global_z, time_until_cp, delta_time], 1).detach()
                else:
                    h_target_cpaw = h_cpaw

                zt  , _, _ = self.posterior(h_target_cpaw)
                zt_p, _, _ = self.prior(h_cpaw)

                if model_mode == 'posterior':
                    h = self.frame_predictor(torch.cat([h, zt, time_until_cp, delta_time], 1))
                elif model_mode == 'prior' or model_mode == 'full':
                    h = self.frame_predictor(torch.cat([h, zt_p, time_until_cp, delta_time], 1))

                x_in = self.decoder([h, skip]).detach()
                gen_seq.append(x_in) # NOTE: gen_seq can append the decoded x_in for comparing with gt
        return gen_seq

    def forward(self, x, start_ix=0, cp_ix=-1):
        """ training """
        if type(x) == tuple: # h36m # NOTE: TODO
            (pose_2d, pose_3d, camera_view) = x
            x = pose_3d

        opt = self.opt
        batch_size = x[0].shape[0]

        # initialize the hidden state
        self.init_hidden(batch_size=batch_size)

        # losses
        mse_loss = 0
        kld_loss = 0
        cpc_loss = 0
        align_loss = 0

        # get global descriptor
        seq_len        = len(x)
        start_ix       = 0
        cp_ix          = seq_len - 1
        x_cp, global_z = self.get_global_descriptor(x, start_ix, cp_ix) # here global_z is h_cp

        # time skipping
        skip_prob = opt.skip_prob

        prev_i = 0
        max_skip_count = seq_len * skip_prob
        skip_count = 0
        probs = np.random.uniform(0, 1, seq_len-1)

        for i in range(1, seq_len):
            #if np.random.uniform(0, 1) <= skip_prob and i > 1 and skip_count < max_skip_count and i != cp_ix:
            #if probs[i-1] <= skip_prob and i >= opt.n_past and skip_count < max_skip_count and i != cp_ix:
            if probs[i-1] <= skip_prob and i >= opt.n_past and skip_count < max_skip_count and i != 1 and i != cp_ix:
                skip_count += 1
                continue

            if i > 1:
                align_loss += self.align_criterion(h[0], h_pred)

            time_until_cp = torch.zeros(batch_size, 1).fill_((cp_ix-i+1)/cp_ix).to(x_cp)
            delta_time = torch.zeros(batch_size, 1).fill_((i-prev_i)/cp_ix).to(x_cp)
            prev_i = i

            h = self.encoder(x[i-1])
            h_target = self.encoder(x[i])[0]
            
            #if opt.last_frame_skip or i < opt.n_past: # original
            if opt.last_frame_skip or i <= opt.n_past:
                h, skip = h
            else:
                h = h[0]

            # cp aware
            h_cpaw        = torch.cat([h, global_z, time_until_cp, delta_time], 1)
            h_target_cpaw = torch.cat([h_target, global_z, time_until_cp, delta_time], 1)

            zt, mu, logvar    = self.posterior(h_target_cpaw)
            zt_p, mu_p, logvar_p = self.prior(h_cpaw)

            h_pred = self.frame_predictor(torch.cat([h, zt, time_until_cp, delta_time], 1))
            x_pred = self.decoder([h_pred, skip])

            # loss
            if i == (cp_ix): # the gen-cp-frame should be exactly as x_cp
                h_pred_p = self.frame_predictor(torch.cat([h, zt_p, time_until_cp, delta_time], 1))
                x_pred_p = self.decoder([h_pred_p, skip])
                cpc_loss = self.mse_criterion(x_pred_p, x_cp)

            mse_loss += self.mse_criterion(x_pred, x[i])
            kld_loss += self.kl_criterion(mu, logvar, mu_p, logvar_p)

        # backward
        # update model without prior
        loss = mse_loss + kld_loss*opt.beta + align_loss*opt.weight_align
        loss.backward(retain_graph=True)
        self.update_model_without_prior()

        # update model with prior due to loss_on_prior
        self.prior.zero_grad()
        prior_loss = kld_loss + cpc_loss*opt.weight_cpc
        prior_loss.backward()
        self.update_prior()

        return mse_loss.data.cpu().numpy()/seq_len, kld_loss.data.cpu().numpy()/seq_len, cpc_loss.data.cpu().numpy()/seq_len, align_loss.data.cpu().numpy()/seq_len

    def update_prior(self):
        self.prior_optimizer.step()

    def update_model_without_prior(self):
        self.frame_predictor_optimizer.step()
        self.posterior_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def update_model(self):
        self.frame_predictor_optimizer.step()
        self.posterior_optimizer.step()
        self.prior_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def save(self, fname, epoch):
        # cannot torch.save with module
        backbone_net = self.opt.backbone_net
        self.opt.backbone_net = 0
        states = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'frame_predictor': self.frame_predictor.state_dict(),
            'posterior': self.posterior.state_dict(),
            'prior': self.prior.state_dict(),
            'encoder_opt': self.encoder_optimizer.state_dict(),
            'decoder_opt': self.decoder_optimizer.state_dict(),
            'frame_predictor_opt': self.frame_predictor_optimizer.state_dict(),
            'posterior_opt': self.posterior_optimizer.state_dict(),
            'prior_opt': self.prior_optimizer.state_dict(),
            'epoch': epoch,
            'opt': self.opt,
            }
        torch.save(states, fname)
        self.opt.backbone_net = backbone_net

    def load(self, pth=None, states=None):
        """ load from pth or states directly """
        if states is None:
            states = torch.load(pth)

        self.encoder.load_state_dict(states['encoder'])
        self.decoder.load_state_dict(states['decoder'])
        self.frame_predictor.load_state_dict(states['frame_predictor'])
        self.posterior.load_state_dict(states['posterior'])
        self.prior.load_state_dict(states['prior'])

        self.encoder_optimizer.load_state_dict(states['encoder_opt'])
        self.decoder_optimizer.load_state_dict(states['decoder_opt'])
        self.frame_predictor_optimizer.load_state_dict(states['frame_predictor_opt'])
        self.posterior_optimizer.load_state_dict(states['posterior_opt'])
        self.prior_optimizer.load_state_dict(states['prior_opt'])

        self.opt = states['opt']
        start_epoch = states['epoch'] + 1

        return start_epoch

        
