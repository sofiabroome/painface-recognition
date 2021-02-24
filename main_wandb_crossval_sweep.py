import pandas as pd
import numpy as np
import wandb
import sys
import re

import test_and_eval
import data_handler
import arg_parser
import helpers
import models
import train

from datetime import datetime
from uuid import uuid4
from data_scripts import make_crossval_commands


# hyperparams = ['batch_size', 'dropout_1', 'kernel_size', 'lr',
#                'nb_lstm_layers', 'nb_lstm_units', 'optimizer']

hyperparams = ['nb_layers_enc', 'nb_heads_enc', 'model_size']


def run():

    print(config_dict)
    dh = data_handler.DataHandler(data_columns=['pain'],  # or e.g., 'observer',
                                  config_dict=config_dict,
                                  all_subjects_df=all_subjects_df)

    f1s = []
    pain_f1s = []
    nopain_f1s = []

    for ind, test_subject in enumerate(test_horses):

        config_dict['job_identifier'] = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        print('Job identifier: ', config_dict['job_identifier'])

        if config_dict['val_mode'] == 'subject':
            val_subjects = make_crossval_commands.get_val(args.dataset_str, test_subject)
        if config_dict['val_mode'] == 'no_val':
            val_subjects = ''
        train_subjects = [x for x in train_horses
                          if x is not test_subject
                          and x not in val_subjects]
        print('Subjects to train on: ', train_subjects)
        print('Subject to validate on: ', val_subjects)
        print('Subject to test on: ', test_subject)
        config_dict['train_subjects'] = train_subjects
        config_dict['test_subjects'] = test_subject

        # Train the model

        model = models.MyModel(config_dict=config_dict)
        if config_dict['inference_only']:
            best_model_path = config_dict['checkpoint']
        if config_dict['train_video_level_features']:
            train_dataset = dh.features_to_dataset(train_subjects, split='train')
            if not config_dict['val_mode'] == 'no_val':
                val_dataset = dh.features_to_dataset(val_subjects, split='val')
            else:
                val_dataset = None
            print('Training on loaded features...')
            # samples = [sample for sample in dataset]
            best_model_path = train.video_level_train(
                model=model.model,
                config_dict=config_dict,
                train_dataset=train_dataset,
                val_dataset=val_dataset)

        if config_dict['do_evaluate']:
            if config_dict['video_level_mode']:
                test_dataset = dh.features_to_dataset([test_subject], split='test')
                test_paths = [sample[3].numpy().tolist() for sample in test_dataset]
                test_steps = len(test_paths)

                classification_report = test_and_eval.evaluate_on_video_level(
                    config_dict=config_dict,
                    model=model,
                    model_path=best_model_path,
                    test_dataset=test_dataset,
                    test_steps=test_steps)
                f1s.append(classification_report['macro avg']['f1-score'])
                nopain_f1s.append(classification_report['NO_PAIN']['f1-score'])
                pain_f1s.append(classification_report['PAIN']['f1-score'])

    avg_f1 = np.mean(f1s)
    avg_nopain_f1 = np.mean(nopain_f1s)
    avg_pain_f1 = np.mean(pain_f1s)

    std_f1 = np.std(f1s)
    std_nopain_f1 = np.std(nopain_f1s)
    std_pain_f1 = np.std(pain_f1s)

    wandb.log({'avg_f1': avg_f1})
    wandb.log({'avg_nopain_f1': avg_nopain_f1})
    wandb.log({'avg_pain_f1': avg_pain_f1})

    wandb.log({'std_f1': std_f1})
    wandb.log({'std_nopain_f1': std_nopain_f1})
    wandb.log({'std_pain_f1': std_pain_f1})


def overwrite_hyperparams_in_config():
    args_dict = vars(args)
    for hp in hyperparams:
        config_dict[hp] = args_dict[hp]

if __name__ == '__main__':

    arg_parser = arg_parser.ArgParser(len(sys.argv))
    args = arg_parser.parse()

    train_horses, test_horses = make_crossval_commands.get_train_test(
        args.dataset_str, avoid_sir_holger=True)

    config_dict_module = helpers.load_module(args.config_file)
    config_dict = config_dict_module.config_dict
    if config_dict['val_mode'] == 'no_val':
        assert (config_dict['train_mode'] == 'low_level'), \
                'no_val requires low level train mode'

    overwrite_hyperparams_in_config()
    config_dict['nb_layers_dec'] = config_dict['nb_layers_enc'] 
    config_dict['nb_heads_dec'] = config_dict['nb_heads_enc'] 
    wandb.init(project='pfr', config=config_dict)

    all_subjects_df = pd.read_csv(args.subjects_overview)
    if args.test_run == 1:
        config_dict['epochs'] = 1
        config_dict['video_nb_epochs'] = 1

    # Run the whole program, from preparing the data to evaluating
    # the model's test performance
    run()
