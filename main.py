import subprocess
import sys

from models.resnet_1d import model_ResNet

def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])

# install('tensorboard')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.SimCLR import SimCLR
from models.linear_evaluation import LinearEvaResNet
from train import trainSimCLR, trainLinearEvalution, trainSimCLR_
# from torchinfo import summary

from utils.dataset import TransDataset
import configs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_dataloader(is_training=True):
    trainSet = TransDataset(configs.filepath_train, is_training=is_training)
    testSet = TransDataset(configs.filepath_test, is_training=is_training)
    trainLoader = DataLoader(
        trainSet,
        batch_size=configs.batchsize,
        shuffle=True,
        drop_last=True, 
        num_workers=8)
    testLoader = DataLoader(
        testSet,
        batch_size=configs.batchsize,
        shuffle=True,
        drop_last=True,
        num_workers=8)
    return trainLoader, testLoader

def create_model(pretrain, load_pretrained = True, freeze_encoder=True):
    if pretrain:
        model = SimCLR(configs.viewmaker_configs, configs.encoder_configs)
        # state_dict = torch.load(configs.save_model_path)
        # model.load_state_dict(state_dict)
    else:
        model = LinearEvaResNet(configs.num_classes, configs.encoder_configs, viewmaker_config=configs.viewmaker_configs, use_viewer=True)
        # model = nn.DataParallel(model)
        if load_pretrained:
            state_dict = torch.load(configs.save_model_path)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model_state = model.state_dict()
            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_state and "encoder" in k}
            print(pretrained_dict.keys())
            model_state.update(pretrained_dict)
            model.load_state_dict(model_state)
            
            try:
                for param in model.view.parameters():
                        param.requires_grad = False
            except:
                pass
            
            if freeze_encoder:
                for param in model.encoder.parameters():
                    param.requires_grad = False
    return model

def main():
    trainLoader, testLoader = create_dataloader(is_training=configs.pretrain)
    # trainSet = TransDataset(configs.filepath_train, is_training=is_training)
    # testSet = TransDataset(configs.filepath_test, is_training=True)
    # model = model_ResNet([2,2,2,2], 
    #                      inchannel=configs.in_channel, 
    #                      num_classes=configs.num_classes).to(device)
    model = create_model(pretrain=configs.pretrain, 
                         load_pretrained=True, 
                         freeze_encoder=False).to(device)
    model = nn.DataParallel(model)
    # summary(model, ((256, 1, 6000), (256, 1, 6000)))
    if configs.pretrain:
        if configs.viewmaker_configs['use_viewmaker']:
            trainSimCLR(model, trainLoader, testLoader, device)
        else:
            trainSimCLR_(model, trainLoader, testLoader, device)
    else:
        trainLinearEvalution(model, trainLoader, testLoader, device)
    
if __name__ == '__main__':
    main()