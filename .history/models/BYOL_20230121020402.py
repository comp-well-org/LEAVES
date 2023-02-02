import torch 
from torch import nn
import torch.nn.functional as F
from models.resnet_1d import model_ResNet
import configs


def MLP(dim, projection_size, hidden_size=1024):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )
    

class encoderWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2) -> None:
        super().__init__()
        self.net = net
        self.layer = layer
        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        
        self.hidden = {}
        self.hook
        
    
class BYOL(nn.Module):
    def __init__(self, net, input_size, hidden_layer, 
                 projection_size=128,
                 projection_hidden_size=1024,
                 moving_average_decay=0.99,
                 use_momentum=True) -> None:
        super().__init__()
        
        self.net = net
        
        self.online_encoder = model_ResNet([2,2,2,2], 
                    inchannel=configs.in_channel, 
                    num_classes=configs.num_classes)