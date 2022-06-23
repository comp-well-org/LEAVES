################################ Data Configs ###############################
filepath_train = '/rdf/data/physionet.org/processed_DA/sleep_edfe/train_EEG.csv'
filepath_test = '/rdf/data/physionet.org/processed_DA/sleep_edfe/test_EEG.csv'

num_classes = 5

batchsize = 1024
LR = 1e-3
epochs = 500
save_path = './experiments/init_run/'
save_model_path = save_path + 'checkpoint_20.pth'
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
    'n_block':8,
    'downsample_gap':2,
    'increasefilter_gap':2,
}