import configs.pixel_means as pixel_means
data_path = '/local_storage/users/sbroome/painface-recognition/'
# data_path = 'data/'

config_dict = {
    # Program components
    'get_raw_sequence_data': False,
    'inference_only': True,
    'fine_tune': False,
    'save_features': False,
    'save_features_per_video': False,
    'video_level_mode': True,
    'train_video_level_features': True,
    'do_evaluate': True,
    'val_mode': 'subject',  # subject | fraction | no_val
    'train_mode': 'low_level',  # keras | low_level
    # Data
    'data_path': data_path,
    'clip_list_pf': 'metadata/videos_overview_missingremoved.csv',
    'clip_list_lps': 'metadata/lps_videos_overview.csv',
    'pf_rgb_path': data_path + 'pf/jpg_128_128_2fps/',
    'lps_rgb_path': data_path + 'lps/jpg_128_128_2fps/',
    'pf_of_path': data_path + 'pf/jpg_128_128_16fps_OF_magnitude_cv2/',
    'lps_of_path': data_path + 'lps/jpg_128_128_16fps_OF_magnitude_cv2_2fpsrate/',
    'pixel_mean': pixel_means.pf_rgb['mean'],
    'pixel_std': pixel_means.pf_rgb['std'],
    # 'checkpoint': 'models/124805_best_model_2stream_5d_add.ckpt',
    'checkpoint': 'models/127575_last_model_video_level_preds_mil_attn.ckpt',
    'save_video_features_folder': 'lps/video_level_features_320dim_zeropad_noresample/',
    # 'train_video_features_folder': 'lps/video_level_features_320dim_resample/',
    # 'val_video_features_folder': 'lps/video_level_features_320dim_noresample/',
    # 'test_video_features_folder': 'lps/video_level_features_320dim_noresample/',
    'train_video_features_folder': 'lps/video_level_features_320dim_zeropad_resample/',
    'val_video_features_folder': 'lps/video_level_features_320dim_zeropad_noresample/',
    'test_video_features_folder': 'lps/video_level_features_320dim_zeropad_noresample/',
    # 'train_video_features_folder': 'lps/video_level_features_320dim_test/',
    # 'val_video_features_folder': 'lps/video_level_features_320dim_test/',
    # 'test_video_features_folder': 'lps/video_level_features_320dim_test/',
    # Model
    'model': '2stream_5d_add',
    'rgb_period': 1,  # Set to 10 if simonyan-like model
    'flow_period': 1,
    'input_width': 128,
    'input_height': 128,
    'color': True,
    'nb_labels': 2,
    'target_names': ['NO_PAIN', 'PAIN'],
    'nb_lstm_units': 32,
    'nb_lstm_layers': 4,
    'kernel_size': 5,
    'dropout_1': 0.25,
    'dropout_2': 0.5,
    'return_last_clstm' : True,
    # Model for video level features
    # 'video_features_model' : 'video_level_preds_attn_network',
    # 'video_features_model' : 'video_fc_model',
    # 'video_loss' : 'cross_entropy',
    'video_features_model' : 'video_level_preds_mil_attn',
    'video_loss' : 'mil',
    'nb_layers' : 1,
    'nb_units' : 32,
    'video_batch_size' : 3,
    'video_pad_length' : 144,
    'video_nb_epochs': 10,
    'video_early_stopping': 50,
    'shuffle_buffer': 150,
    'k_mil_loss': 0.15,
    'tv_weight_pain': 10,
    'tv_weight_nopain': 100,
    # Parameters for functional API C-LSTM
    'kernel_regularizer' : None,
    'padding_clstm' : 'valid',
    'strides_clstm' : (1,1),
    'dropout_clstm' : 0.0,
    'pooling_method' : 'max',
    'return_sequences' : [True, True, True, True],
    'only_last_element_for_fc' : 'no',
    # Training
    'optimizer': 'adadelta',
    'lr': 0.001,
    'nb_epochs': 2,
    'early_stopping': 15,
    'round_to_batch': True,
    'seq_length': 10,
    'seq_stride': 10,
    'nb_workers': 1,
    'batch_size': 8,
    'nb_input_dims': 5,
    'val_fraction_value': 0.0,
    'monitor': 'val_binary_accuracy',
    'monitor_mode': 'max',
    'data_type': 'rgb',
    'aug_flip': 0,
    'aug_crop': 0,
    'aug_light': 0,
    'print_loss_every': 300,
    'resample_start_fraction_of_seq_length': 0.5,
    # Temporal mask things
    'normalization_mode': 'sequence',  # 'frame' | 'sequence'
    'temporal_mask_type': 'freeze',
    'nb_iterations_graddescent': 500,
    'focus_type': 'guessed',
    'lambda_1': 0.01,
    'lambda_2': 0.02,
    'tv_norm_p': 3,
    'tv_norm_q': 3,
    'verbose': True,
    'do_gradcam': True}
