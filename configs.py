pretrain = True ## Ture if in contrastive learning, False if for down-stream tasks


################################ Data Configs ###############################
import os
# data_dir_aws = os.environ["SM_CHANNEL_TRAINING"]
data_dir_aws = "/home/hanyu/data/ptbxl/100hz"
# data_dir_aws = "/rdf/data/PTB-XL/"

filepath_train = os.path.join(data_dir_aws, "train_dict_12lead.npy")
filepath_test = os.path.join(data_dir_aws, "test_dict_12lead.npy")

num_classes = 5

batchsize = 192
LR = 1e-3
epochs = 100
# save_path = '/opt/ml/model/'

save_path = './experiments/ptbxl/leaves/byol_v2/' 
save_model_path = save_path + 'checkpoint_100.pth' ## path of saved model in contrastive learning, model will load weights for down-stream tasks
################################ aug Configs ################################
noise_sigma = 0.5
warp_sigma = 0.5
use_attention = False
################################ Model Configs ################################
in_channel = 12
projection_size = 128
projection_hidden_size = 1024
################################ Dual Modal Configs ################################
in_channel1 = 1
in_channel2 = 1
projection_size = 128
projection_hidden_size = 1024
dual_modal = False

supResolution = 10

leaves_configs = {
    'framework': 'byol',
    'use_leaves' : True,
    'num_channels' : in_channel,
    'view_bound_magnitude' : 0.05,
    'clamp' : True 
}

# encoder_configs = {
#     'in_channels':in_channel,
#     'base_filters':32, # 64 for ResNet1D, 352 for ResNeXt1D
#     'kernel_size':10, 
#     'stride':2,
#     'n_block':18,
#     'downsample_gap':2,
#     'increasefilter_gap':4,
# }