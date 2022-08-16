################################ Data Configs ###############################
filepath_train = '/rdf/data/physionet.org/processed_DA/sleep_apnea/sleep_apnea_train.csv'
filepath_test = '/rdf/data/physionet.org/processed_DA/sleep_apnea/sleep_apnea_test.csv'

num_classes = 2

batchsize = 128
LR = 5e-3
epochs = 500
save_path = './experiments/sleep_apnea/normalization/'
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