pretrain = True


################################ Data Configs ###############################
import os
# data_dir_aws = os.environ["SM_CHANNEL_TRAINING"]
data_dir_aws = "/rdf/data/physionet.org/processed_DA/PAMAP2_Dataset/"
# data_dir_aws = "/rdf/data/PTB-XL/"

filepath_train = os.path.join(data_dir_aws, "dataset_train.npy")
filepath_test = os.path.join(data_dir_aws, "dataset_test.npy")

num_classes = 11

batchsize = 128
LR = 1e-3
epochs = 200
# save_path = '/opt/ml/model/'

save_path = '/home/hy29/rdf/viewmaker_physiological/experiments/PAMAP2/init_run/'
save_model_path = save_path + 'checkpoint_100.pth'
################################ aug Configs ################################
noise_sigma = 0.3
warp_sigma = 0.2
################################ Model Configs ################################
in_channel = 52

viewmaker_configs = {
    'use_viewmaker' : False,
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