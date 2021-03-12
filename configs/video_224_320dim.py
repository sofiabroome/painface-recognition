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
    'tfrecords': True,
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
    'checkpoint': 'models/206260_last_model_video_level_preds_attn_gru_network.ckpt',
    'save_video_features_folder': 'lps/video_level_features_320dim_noresample/',
    'tfr_file': 'tfrecords/pflps_video_level_features_169250_320dim_zeropad266_noresample/videofeats_2stream_flat',
    'minor_class_weight': 62/38,
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
    'nb_lstm_layers': 4,
    'kernel_size': 5,
    'dropout_1': 0,
    'dropout_2': 0.5,
    'return_last_clstm' : True,
    'model_size': 64,
    # Model for videolevel features
    # 'video_features_model' : 'video_level_network',
    # 'video_features_model' : 'video_fc_model',
    # 'video_features_model' : 'video_conv_seq_model',
    # 'video_loss' : 'cross_entropy',
    'video_features_model' : 'video_level_preds_attn_gru_network',
    # 'video_features_model': 'transformer',
    'merge_attn': 'add',
    'mil_version' : 'mil_pain',  # 'mil_pain'|'mil_both'
    'label_smoothing': 0,
    # 'video_features_model' : 'video_level_mil_feats_preds',
    'video_loss' : 'cross_entropy_scores',  # 'mil' | 'mil_ce' | 'cross_entropy' | 'pseudo_labels' | 'cross_entropy_scores'
    # 'video_loss' : 'pseudo_labels',  # 'mil' | 'mil_ce' | 'cross_entropy' | 'pseudo_labels' | 'cross_entropy_scores'
    'nb_layers' : 1,
    'nb_units_1' : 64,
    'nb_units_2' : 32,
    'feature_dim': 320,
    # 'feature_dim': 320,
    'video_batch_size_train' : 20,
    'video_batch_size_test' : 1,
    'video_pad_length' : 266,
    'video_nb_epochs': 1000,
    'video_early_stopping': 200,
    'shuffle_buffer': 150,
    'k_mil_fraction': 0.5,  # This is to be updated using below params
    'k_mil_fraction_start': 0.01,
    'k_mil_fraction_end': 0.01,
    'k_mil_fraction_decrement_step': 0.05,
    'k_mil_fraction_nb_epochs_to_decrease': 1,
    'tv_weight_pain': 0,
    'tv_weight_nopain': 0.1,
    'do_minor_class_weight': True,
    'l1_nopain': True,
    'l2_weight': 0,
    'mc_dropout_samples': 1,
    # Transformer settings
    'model_size': 32,
    'nb_heads_enc': 8,
    'nb_heads_dec': 8,
    'nb_layers_enc': 8,
    'nb_layers_dec': 8,
    # Parameters for functional API C-LSTM
    'kernel_regularizer' : None,
    'padding_clstm' : 'valid',
    'strides_clstm' : (1,1),
    'dropout_clstm' : 0.0,
    'pooling_method' : 'max',
    'return_sequences' : [True, True, True, True],
    'only_last_element_for_fc' : 'no',
    # Training
    'optimizer': 'rmsprop_lrdecay',  # 'rmsprop_lrdecay' | 'rmsprop' | 'adam' | 'adam_warmup_decay' | 'SGD'
    'lr': 0.00001,
    'nb_epochs': 2,
    'early_stopping': 15,
    'round_to_batch': True,
    'seq_length': 10,
    'seq_stride': 10,
    'nb_workers': 1,
    'batch_size': 8,
    'nb_input_dims': 5,
    'val_fraction_value': 0.0,
    'monitor': 'val_f1',  # 'val_binary_accuracy' | 'val_f1'
    'monitor_mode': 'max',
    'data_type': 'rgb',
    'aug_flip': 0,
    'aug_crop': 0,
    'aug_light': 0,
    'print_loss_every': 100000,
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
