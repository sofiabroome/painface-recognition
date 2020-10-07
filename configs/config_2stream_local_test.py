import configs.pixel_means as pixel_means
data_path = 'data/'

config_dict = {
    'train_dataset': 'pf',
    'test_dataset': 'lps',
    'clip_list_pf': 'metadata/videos_overview_missingremoved.csv',
    'clip_list_lps': 'metadata/lps_videos_overview.csv',
    'pf_rgb_path': data_path + 'pf/jpg_128_128_2fps/',
    'lps_rgb_path': data_path + 'lps/jpg_128_128_2fps/',
    'pf_of_path': data_path + 'pf/jpg_128_128_16fps_OF_magnitude_cv2/',
    'lps_of_path': data_path + 'lps/jpg_128_128_16fps_OF_magnitude_cv2_2fpsrate/',
    'pixel_mean': pixel_means.pf_rgb['mean'],
    'pixel_std': pixel_means.pf_rgb['std'],
    'model': '2stream_5d_add',
    'rgb_period': 1,  # Set to 10 if simonyan-like model
    'flow_period': 1,
    'input_width': 128,
    'input_height': 128,
    'color': True,
    'nb_labels': 2,
    'target_names': ['NO_PAIN', 'PAIN'],
    'nb_lstm_units': 2,
    'kernel_size': 3,
    'dropout_1': 0.7,
    'dropout_2': 0.7,
    'nb_epochs': 2,
    'early_stopping': 15,
    'optimizer': 'adadelta',
    'lr': 0.001,
    'round_to_batch': True,
    'seq_length': 10,
    'seq_stride': 10,
    'nb_workers': 1,
    'batch_size': 8,
    'nb_input_dims': 5,
    'val_mode': 'fraction',  # subject | fraction | no_val
    'val_fraction_value': 0.5,
    'monitor': 'val_binary_accuracy',
    'monitor_mode': 'max',
    'data_type': 'rgb',
    'nb_lstm_layers': 2,
    'aug_flip': 0,
    'aug_crop': 0,
    'aug_light': 0,
    'do_evaluate': True,
    'train_mode': 'keras',
    'print_loss_every': 100}
