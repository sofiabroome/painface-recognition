import configs.pixel_means as pixel_means
data_path = '/local_storage/users/sbroome/painface-recognition/'

config_dict = {
    # Program components
    'get_raw_sequence_data': True,
    'inference_only': True,
    'fine_tune': False,
    'save_features': True,
    'save_features_per_video': False,
    'video_level_mode': False,
    'train_video_level_features': False,
    'do_evaluate': False,
    'val_mode': 'subject',  # subject | fraction | no_val
    'train_mode': 'low_level',
    # 'train_mode': 'keras',
    # Data
    'clip_list_pf': 'metadata/videos_overview_missingremoved.csv',
    'clip_list_lps': 'metadata/lps_videos_overview.csv',
    'pf_rgb_path': data_path + 'pf/jpg_224_224_2fps/',
    'lps_rgb_path': data_path + 'lps/jpg_224_224_2fps/',
    'pf_of_path': data_path + 'pf/jpg_224_224_25fps_OF_magnitude_cv2_2fpsrate/',
    'lps_of_path': data_path + 'lps/jpg_224_224_25fps_OF_magnitude_cv2_2fpsrate/',
    'pixel_mean': pixel_means.pf_rgb['mean'],
    'pixel_std': pixel_means.pf_rgb['std'],
    'checkpoint': 'models/pf224_115epochs/169250_last_model_2stream_5d_add.ckpt',
    'save_clip_feats_id': 'lps_saved_features_320dims',  # With 18 first chars of ckpt.
    # Model
    'model': '2stream_5d_add',
    'rgb_period': 1,  # Set to 10 if simonyan-like model
    'flow_period': 1,
    'input_width': 224,
    'input_height': 224,
    'color': True,
    'nb_labels': 2,
    'target_names': ['NO_PAIN', 'PAIN'],
    'nb_lstm_units': 32,
    'kernel_size': 5,
    'dropout_1': 0.25,
    'dropout_2': 0.5,
    # Model for video level features
    'video_features_model' : 'video_level_network',
    'nb_layers' : 1,
    'nb_units' : 16,
    'video_batch_size' : 3,
    # 'video_pad_length' : 150,
    'video_nb_epochs': 100,
    'video_early_stopping': 15,
    'shuffle_buffer': 150,
    # Training
    'nb_epochs': 200,
    'early_stopping': 50,
    'optimizer': 'adadelta',
    'lr': 0.001,
    'round_to_batch': True,
    'seq_length': 10,
    'seq_stride': 10,
    'nb_workers': 1,
    'batch_size': 2,
    'nb_input_dims': 5,
    'val_fraction_value': 0.0,
    'monitor': 'val_binary_accuracy',
    'monitor_mode': 'max',
    'data_type': 'rgb',
    'nb_lstm_layers': 4,
    'aug_flip': 1,
    'aug_crop': 0,
    'aug_light': 0,
    'print_loss_every': 5000,
    'resample_start_fraction_of_seq_length': 0.5,
    # Parameters for functional API C-LSTM
    'kernel_regularizer' : None,
    'padding_clstm' : 'valid',
    'strides_clstm' : (1,1),
    'dropout_clstm' : 0.0,
    'pooling_method' : 'max',
    'return_sequences' : [True, True, True, True],
    'only_last_element_for_fc' : 'no',
    'return_last_clstm' : True}
