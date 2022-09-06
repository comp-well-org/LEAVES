pretrain = True


################################ Data Configs ###############################
import os
data_dir_aws = os.environ["SM_CHANNEL_TRAINING"]

filepath_train = os.path.join(data_dir_aws, "train_EEG.csv")
filepath_test = os.path.join(data_dir_aws, "test_EEG.csv")

num_classes = 2

batchsize = 192
LR = 1e-3
epochs = 100
save_path = '/opt/ml/model/'
save_model_path = save_path + 'checkpoint_120.pth'
################################ Model Configs ################################
in_channel = 1

viewmaker_configs = {
    'use_viewmaker' : True,
    'num_channels' : in_channel,
    'view_bound_magnitude' : 0.05,
    'clamp' : True 
}

encoder_configs = {
    'in_channels':in_channel,
    'base_filters':32, # 64 for ResNet1D, 352 for ResNeXt1D
    'kernel_size':10, 
    'stride':2,
    'n_block':18,
    'downsample_gap':2,
    'increasefilter_gap':4,
}