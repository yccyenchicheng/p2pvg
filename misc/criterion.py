import torch
import torch.nn as nn
#import pytorch_ssim

class KLCriterion(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt

    def forward(self, mu1, logvar1, mu2, logvar2):
        """KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2))"""
        sigma1 = logvar1.mul(0.5).exp() 
        sigma2 = logvar2.mul(0.5).exp() 
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        return kld.sum() / self.opt.batch_size

class SmoothMSE(nn.Module):
    def __init__(self, opt=None, threshold=0.001):
        super().__init__()
        self.opt = opt
        self.threshold = threshold

    def forward(self, x1, x2):
        _, c, h, w = x1.shape
        mse = ((x1 - x2) ** 2).clamp(min=self.threshold)
        return mse.mean()
    