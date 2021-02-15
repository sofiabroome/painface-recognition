config_dict = {
    # Training
    'k_mil_fraction': 0.05,
    'tv_weight_nopain': 0,
    'tv_weight_pain': 0,
    'l1_nopain': False,
    'nb_labels': 2,
    'lr': 0.00001,
    'epochs': 500,
    'batch_size': 20,
    'val_batch_size': 1,
    # Model
    # 'model_name': 'gru_attention',  # 'gru'|'gru_attention'|'dense'|'id'
    'model_name': 'transformer',  # 'transformer|gru'|'gru_attention'|'dense'|'id'
    # 'model_name': 'gru',  # 'gru'|'gru_attention'|'dense'|'id'
    # 'layers': [2],
    'layers': [32, 2],
    # Transformer settings
    'model_size': 8,
    'nb_layers_enc': 1,
    'nb_layers_dec': 1,
    'nb_heads_enc': 1,
    'nb_heads_dec': 1,
    # Data
    'feature_dim': 1,
    'video_pad_length': 266,
    'base_level': 1,
    'max_intensity_pain': 20,
    'max_intensity_nopain': 1,
    'max_length_pain': 2,
    'max_length_nopain': 2,
    'min_events_pain': 3,
    'nb_events_pain': 10,
    'min_events_nopain': 0,
    'nb_events_nopain': 0
}
