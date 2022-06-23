from re import I
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.SimCLR import SimCLR
from models.linear_evaluation import LinearEvaResNet
from train import trainSimCLR, trainLinearEvalution
from torchinfo import summary

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
        drop_last=True)
    testLoader = DataLoader(
        testSet,
        batch_size=configs.batchsize,
        shuffle=True,
        drop_last=True)
    return trainLoader, testLoader

def create_model(pretrain, freeze_encoder=True):
    if pretrain:
        model = SimCLR(configs.viewmaker_configs, configs.encoder_configs)
    else:
        model = LinearEvaResNet(configs.num_classes, configs.encoder_configs)
        state_dict = torch.load(configs.save_model_path)
        model_state = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_state}
        model_state.update(pretrained_dict)
        model.load_state_dict(model_state)
        if freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False
    return model

def main():
    trainLoader, testLoader = create_dataloader(is_training=False)
    model = create_model(pretrain=False).to(device)
    # summary(model, ((256, 1, 3000), (256, 1, 3000)))
    # trainSimCLR(model, trainLoader, testLoader, device)
    trainLinearEvalution(model, trainLoader, testLoader, device)
    
    
    
if __name__ == '__main__':
    main()
    print(1)
