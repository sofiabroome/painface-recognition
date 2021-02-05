import configs.pixel_means as pixel_means

# data_path = '/local_storage/users/sbroome/painface-recognition/'
data_path = '../data/'

config_dict = {
    # Program components
    'get_raw_sequence_data': True,
    'inference_only': False,
    'fine_tune': False,
    'save_features': False,
    'save_features_per_video': False,
    'video_level_mode': False,
    'train_video_level_features': False,
    'do_evaluate': False,
    # Data
    'clip_list_pf': 'metadata/videos_overview_missingremoved.csv',
    'clip_list_lps': 'metadata/lps_videos_overview.csv',
    'pf_rgb_path': data_path + 'pf/jpg_128_128_2fps/',
    'lps_rgb_path': data_path + 'lps/jpg_128_128_2fps/',
    'pf_of_path': data_path + 'pf/jpg_128_128_16fps_OF_magnitude_cv2/',
    'lps_of_path': data_path + 'lps/jpg_128_128_16fps_OF_magnitude_cv2_2fpsrate/',
    # 'data_df_path': data_path + 'pf/' + 'jpg_128_128_16fps_OF_magnitude_cv2/horse_3.csv',
    'data_df_path': data_path + 'lps/interpretability_results/top_k/A_20190104_IND3_STA_2/top_3_pain.csv',
    'pixel_mean': pixel_means.pf_rgb['mean'],
    'pixel_std': pixel_means.pf_rgb['std'],
    # 'output_folder': data_path + 'results/interpretability_results/',
    'output_folder': data_path + 'lps/interpretability_results/top_k/A_20190104_IND3_STA_2/',
    # 'checkpoint' : '../models/BEST_MODEL_2stream_5d_add_117332.h5', # 2stream
    # 'checkpoint' : '../models/best_model_2stream_5d_add_testhej.ckpt',
    'checkpoint': '../models/132766_best_model_2stream_5d_add.ckpt',
    'model': '2stream_5d_add',
    # 'checkpoint' : '../models/BEST_MODEL_convolutional_LSTM_116306.h5',  # 1stream
    # 'checkpoint': '../models/best_model_convolutional_LSTM_testhej.ckpt',  # 1stream
    # 'model': 'convolutional_LSTM',
    'rgb_period': 1,  # Set to 10 if simonyan-like model
    'flow_period': 1,
    'input_width': 128,
    'input_height': 128,
    'color': True,
    'nb_labels': 2,
    'target_names': ['NO_PAIN', 'PAIN'],
    'nb_lstm_units': 32,
    'kernel_size': 5,
    'dropout_1': 0.25,
    'dropout_2': 0.5,
    'nb_epochs': 100,
    'early_stopping': 15,
    'optimizer': 'adadelta',
    'lr': 0.1,
    'round_to_batch': True,
    'seq_length': 10,
    'seq_stride': 10,
    'nb_workers': 1,
    'batch_size': 1,
    'nb_input_dims': 5,
    'val_mode': 'no_val',  # subject | fraction | no_val
    'val_fraction_value': 0.001,
    'monitor': 'val_binary_accuracy',
    'monitor_mode': 'max',
    'data_type': 'rgb',
    'nb_lstm_layers': 4,
    'aug_flip': 0,
    'aug_crop': 0,
    'aug_light': 0,
    'do_evaluate': True,
    'train_mode': 'keras',
    'print_loss_every': 100,
    'padding_clstm': 'same',
    'pooling_method': 'max',  # avg | max
    'only_last_element_for_fc': True,
    'stride_clstm': 1,
    'dropout_clstm': 0.0,
    'kernel_regularizer': 0.0,
    'return_sequences': '[True, True, True, False]',
    'return_last_clstm': True,
    'resample_start_fraction_of_seq_length': 0.5,
    # Temporal mask things
    'inference_only': True,
    'normalization_mode': 'sequence',  # 'frame' | 'sequence'
    'temporal_mask_type': 'freeze',
    'nb_iterations_graddescent': 25,
    'focus_type': 'guessed',
    'lambda_1': 1,
    'lambda_2': 0.02,
    'tv_norm_p': 3,
    'tv_norm_q': 3,
    'verbose': True,
    'do_gradcam': True}
