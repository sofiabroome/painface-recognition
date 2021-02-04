config_dict = {
    # Training
    'k_mil_fraction': 0.05,
    'tv_weight_nopain': 0.1,
    'tv_weight_pain': 0,
    'nb_labels': 2,
    'lr': 0.00001,
    'epochs': 500,
    'batch_size': 20,
    'val_batch_size': 1,
    # Model
    'model_name': 'gru',  # 'gru'|'dense'|'id'
    'layers': [2],
    # 'layers': [32, 2],
    # Data
    'T': 266,
    'base_level': 1,
    'max_intensity_pain': 2000,
    'max_intensity_nopain': 1,
    'max_length_pain': 2,
    'max_length_nopain': 2,
    'min_events_pain': 30,
    'nb_events_pain': 50,
    'min_events_nopain': 0,
    'nb_events_nopain': 3
}
