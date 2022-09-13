pretrain = True


################################ Data Configs ###############################
import os
# data_dir_aws = os.environ["SM_CHANNEL_TRAINING"]
data_dir_aws = "/rdf/data/PTB-XL/"

filepath_train = os.path.join(data_dir_aws, "train_dict_12lead.npy")
filepath_test = os.path.join(data_dir_aws, "test_dict_12lead.npy")

num_classes = 5

batchsize = 128
LR = 1e-3
epochs = 200
# save_path = '/opt/ml/model/'

save_path = '/home/hy29/rdf/viewmaker_physiological/experiments/ptbxl/init_run/'
save_model_path = save_path + 'checkpoint_70.pth'
################################ Model Configs ################################
in_channel = 12

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