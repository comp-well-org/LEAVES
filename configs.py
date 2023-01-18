pretrain = False ## Ture if in contrastive learning, False if for down-stream tasks


################################ Data Configs ###############################
import os
# data_dir_aws = os.environ["SM_CHANNEL_TRAINING"]
data_dir_aws = "/rdf/data/physionet.org/processed_DA/sleep_apnea/"
# data_dir_aws = "/rdf/data/PTB-XL/"

filepath_train = os.path.join(data_dir_aws, "sleep_apnea_train.csv")
filepath_test = os.path.join(data_dir_aws, "sleep_apnea_test.csv")

num_classes = 2

batchsize = 128
LR = 1e-3
epochs = 50
# save_path = '/opt/ml/model/'

save_path = '/home/hy29/rdf/viewmaker_physiological/experiments/sleep_apnea/baseline/sigma01/' 
save_model_path = save_path + 'checkpoint_100.pth' ## path of saved model in contrastive learning, model will load weights for down-stream tasks
################################ aug Configs ################################
noise_sigma = 0.5
warp_sigma = 0.5
use_attention = False
################################ Model Configs ################################
in_channel = 1

viewmaker_configs = {
    'use_viewmaker' : False,
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