# The implementation of BYOL is based on https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py

import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from models.auto_aug import autoAUG
from models.resnet_1d import model_ResNet, model_ResNet_dualmodal
import configs

# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )
    
# main class

class BYOL(nn.Module):
    def __init__(
        self,
        leaves_config,
        encoder_config,
        moving_average_decay = 0.99,
        use_momentum = True,
        dual_modal=configs.dual_modal
    ):
        super().__init__()
        self.dual_modal = dual_modal
        self.leaves_config = leaves_config
        if self.leaves_config['use_leaves']:
            self.view = autoAUG(num_channel = configs.in_channel)
            self.view2 = autoAUG(num_channel = configs.in_channel)
        self.encoder = self.create_encoder()
        if not dual_modal:
            self.fc = nn.Linear(512, 16)
        else:
            self.fc = nn.Linear(1024, 16)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(configs.projection_size, configs.projection_size, configs.projection_hidden_size)

    def create_encoder(self):
        encoder = model_ResNet([2,2,2,2], 
                    inchannel=configs.in_channel, 
                    num_classes=configs.num_classes)
        if self.dual_modal:
            encoder = model_ResNet_dualmodal([2,2,2,2], 
                        inchannel1=configs.in_channel1, 
                        inchannel2=configs.in_channel2, 
                        num_classes=configs.num_classes)
            
        return encoder
    
    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x1,
        x2,
    ):

        if self.dual_modal:
            if self.leaves_config['use_leaves']:
                x1_1 = self.view(x1)
                x1_2 = self.view(copy.deepcopy(x1))
                x2_1 = self.view2(x2)
                x2_2 = self.view2(copy.deepcopy(x2))
            
            online_proj_one = self.fc(self.encoder(x1_1, x2_1))
            online_proj_two = self.fc(self.encoder(x1_2, x2_2))
        else:
            if self.leaves_config['use_leaves']:
                x1 = self.view(x1)
                x2 = self.view(x2)
            
            online_proj_one = self.fc(self.encoder(x1))
            online_proj_two = self.fc(self.encoder(x2))

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            if self.dual_modal:
                target_proj_one = self.fc(target_encoder(x1_1, x2_1))
                target_proj_two = self.fc(target_encoder(x1_2, x2_2))
            else:
                target_proj_one = self.fc(target_encoder(x1))
                target_proj_two = self.fc(target_encoder(x2))
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_proj_one, target_proj_two.detach())
        loss_two = loss_fn(online_proj_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()