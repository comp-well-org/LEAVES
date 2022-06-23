import torch
import torch.nn as nn
from torch.nn import functional as F
from models.resnet import ResNetEncoder

class LinearEvaResNet(nn.Module):
    def __init__(self, num_classes, encoder_config):
        super().__init__()
        self.num_classes = num_classes
        
        self.encoder = self.create_encoder(encoder_config)
        self.fc = nn.Linear(256, 1024)
        self.dense = nn.Linear(1024, self.num_classes)
        self.dropout = nn.Dropout(0.5)
                
        
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
        output = self.encoder(x)
        output = output.view(batch_size, -1)
        output = self.dropout(output)
        output = self.fc(output)
        output = self.dense(output)
        
        return output