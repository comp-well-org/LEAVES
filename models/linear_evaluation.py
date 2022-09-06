from logging import error
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.resnet import ResNetEncoder
# from models.viewmaker import ViewMaker

class LinearEvaResNet(nn.Module):
    def __init__(self, num_classes, encoder_config, viewmaker_config=None, use_viewer=False):
        super().__init__()
        self.num_classes = num_classes
        # self.use_viewer = use_viewer
        # if use_viewer:
        #     if viewmaker_config==None:
        #         raise Exception("Please specify viewmaker configuration if you want to use view maker!")
        #     self.view = self.create_viewmaker(viewmaker_config)
        
        self.encoder = self.create_encoder(encoder_config)
        self.fc = nn.Linear(512, 1024)
        self.dense1 = nn.Linear(1024, 512)
        self.dense2 = nn.Linear(512, self.num_classes)
        self.dropout = nn.Dropout(0.5)
    
    # def create_viewmaker(self, viewmaker_config):
    #     view_model = ViewMaker(num_channels = viewmaker_config['num_channels'],
    #                            distortion_budget = viewmaker_config['view_bound_magnitude'],
    #                            clamp = viewmaker_config['clamp'])
    #     return view_model
        
    def create_encoder(self, encoder_config):
        encoder = ResNetEncoder(
                        in_channels=encoder_config['in_channels'], 
                        base_filters=encoder_config['base_filters'],
                        kernel_size=encoder_config['kernel_size'], 
                        stride=encoder_config['stride'], 
                        groups=1, 
                        n_block=encoder_config['n_block'], 
                        downsample_gap=encoder_config['downsample_gap'], 
                        increasefilter_gap=encoder_config['increasefilter_gap'], 
                        use_do=True)
        return encoder
    
    def forward(self, x):
        batch_size = x.size(0)
        if self.use_viewer:
            output = self.view(x)
        output = self.encoder(x)
        output = output.view(batch_size, -1)
        output = self.dropout(output)
        output = self.fc(output)
        output = self.dense1(output)
        output = self.dense2(output)
        
        return output