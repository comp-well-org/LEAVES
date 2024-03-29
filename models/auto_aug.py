from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.differentiable_augs import jitter, scaling, rotation, time_distortion, permutation, magnitude_warp, freq_depress
import configs

class augAttn(nn.Module):
    def __init__(self, channels, custom_init=True):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels=channels, out_channels=20, kernel_size=1, stride=1,
                                   padding=0)
        self.conv_2 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=1, stride=1,
                                   padding=0)
        self.conv_3 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=1, stride=1,
                                   padding=0)
        self.conv_4 = nn.Conv1d(in_channels=20, out_channels=channels, kernel_size=1, stride=1,
                                   padding=0)
        if custom_init:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    m.weight.data.normal_(0.0, 1.0)
                    m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.sigmoid(self.conv_4(x))

        return x

class autoAUG(nn.Module):
    def __init__(self, num_channel, jitter_sigma=0.03, scaling_sigma=0.03):
        super().__init__()
        self.jitter_sigma = nn.Parameter(jitter_sigma * torch.ones(1), requires_grad=True)
        self.jitter = jitter

        self.scaling_sigma = nn.Parameter(scaling_sigma * torch.ones(1), requires_grad=True)
        self.scaling = scaling

        self.rotation_prob = nn.Parameter((1 - 0.1) * torch.ones(num_channel), requires_grad=True)
        # self.rotation_prob = self.rotation_prob.repeat(configs.batchsize, 1)
        self.rotation = rotation

        self.mixture_weights = nn.Parameter(torch.ones(5), requires_grad=True)
        self.nromal_mean = nn.Parameter(torch.Tensor([-0.5, -0.25, 0.0, 0.25, 0.5]), requires_grad=True)
        self.nromal_sigma = nn.Parameter(torch.Tensor([0.7, 0.7, 0.7, 0.7, 0.7]), requires_grad=True)
        self.timeDis = time_distortion

        self.permuation_seg = nn.Parameter(5 * torch.ones(1), requires_grad=True)
        self.permutation = permutation

        self.magW_sigma = nn.Parameter(0.1 * torch.ones(1), requires_grad=True)
        self.magW = magnitude_warp
        
        self.freq_mask_sigma = nn.Parameter(1 * torch.ones(1), requires_grad=True)
        self.freq_depress = freq_depress

        self.augAttn = augAttn(channels=configs.in_channel)
        self.params = [self.jitter_sigma, self.scaling_sigma, self.rotation_prob, self.mixture_weights,
                       self.nromal_mean, self.nromal_sigma, self.freq_mask_sigma, self.permuation_seg, self.magW_sigma]

        self.e = 1e-5

    def normalization(self, x):
        x -= x.min(2, keepdim=True)[0]
        x /= (x.max(2, keepdim=True)[0] + 0.00000001)
        return x
    
    def forward(self, x): 
        if configs.use_attention:
            attn = self.augAttn(x)
        else:
            attn = None
        x = self.jitter(x, 0.05 * torch.sigmoid(self.jitter_sigma) + self.e, attention=attn)
        x = self.scaling(x, 0.05 * torch.sigmoid(self.scaling_sigma) + self.e)
        # x = self.rotation(x, torch.sigmoid(self.rotation_prob).repeat(x.size(0), 1))
        x = self.freq_depress(x, self.freq_mask_sigma).real
        x = self.timeDis(x, self.mixture_weights, self.nromal_mean, F.relu(self.nromal_sigma) + self.e)
        x = self.permutation(x, self.permuation_seg)
        x = self.magW(x, 0.05 * torch.sigmoid(self.magW_sigma) + self.e)
        x = self.normalization(x)
        return x