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


# Prettyprint for dataframes with long values (filenames).
pd.set_option('max_colwidth', 800)


def run():

    dh = data_handler.DataHandler(data_columns=['pain'],  # or e.g., 'observer',
                                  config_dict=config_dict,
                                  all_subjects_df=all_subjects_df)

    if config_dict['get_raw_sequence_data']:

        train_sequence_dfs, val_sequence_dfs, test_sequence_dfs = dh.get_data_indices(args)

        test_sequence_dfs = dh.round_to_batch_size(test_sequence_dfs)

        train_dataset, val_dataset, test_dataset = dh.get_datasets(
            df_train=train_sequence_dfs,
            df_val=val_sequence_dfs,
            df_test=test_sequence_dfs)

        train_steps = int(len(train_sequence_dfs)/config_dict['batch_size'])

        test_steps = int(len(test_sequence_dfs)/config_dict['batch_size'])
        test_labels, test_paths = dh.get_y_batches_paths_from_dfs(test_sequence_dfs)

        if config_dict['val_mode'] == 'no_val':
            val_steps = 0
        else:
            val_steps = int(len(val_sequence_dfs)/config_dict['batch_size'])

        if args.test_run == 1:
            config_dict['nb_epochs'] = 1
            train_steps = 2
            val_steps = 2
            test_steps = 40
            test_labels = test_labels[:test_steps]
            test_paths = test_paths[:test_steps*config_dict['batch_size']]

    # Train the model

    model = models.MyModel(config_dict=config_dict)
    if config_dict['inference_only']:
        best_model_path = config_dict['checkpoint']
    else:
        best_model_path = train.train(model_instance=model,
                                      config_dict=config_dict,
                                      train_steps=train_steps,
                                      val_steps=val_steps,
                                      train_dataset=train_dataset,
                                      val_dataset=val_dataset)

    if config_dict['save_features']:
        test_dataset = dh.get_dataset(test_sequence_dfs, train=False)
        train.save_features(model.model, config_dict,
                            steps=test_steps, dataset=test_dataset)

    if config_dict['save_features_per_video']:
        f_path = config_dict['data_path'] + 'lps/' + config_dict['checkpoint'][7:18] + '_saved_features_20480dims.npz'
        features = np.load(f_path, allow_pickle=True)
        dh.prepare_video_features(features, zero_pad=True)

    if config_dict['train_video_level_features']:
        train_dataset = dh.features_to_dataset(train_subjects, split='train')
        val_dataset = dh.features_to_dataset(val_subjects, split='val')
        print('Training on loaded features...')
        # samples = [sample for sample in dataset]
        best_model_path = train.video_level_train(
            model=model.model,
            config_dict=config_dict,
            train_dataset=train_dataset,
            val_dataset=val_dataset)

    if config_dict['do_evaluate']:
        if config_dict['video_level_mode']:
            test_dataset = dh.features_to_dataset(test_subjects, split='test')
            test_paths = [sample[3].numpy().tolist() for sample in test_dataset]
            test_steps = len(test_paths)
            test_paths = np.array(test_paths, dtype=object)
            test_labels = np.array([sample[2].numpy().tolist() for sample in test_dataset])

            test_and_eval.evaluate_on_video_level(
                config_dict=config_dict,
                model=model,
                model_path=best_model_path,
                test_dataset=test_dataset,
                test_steps=test_steps)
        else:
            test_and_eval.run_evaluation(
                config_dict=config_dict,
                model=model,
                model_path=best_model_path,
                test_dataset=test_dataset,
                test_steps=test_steps,
                y_batches=test_labels,
                y_paths=test_paths)

if __name__ == '__main__':

    arg_parser = arg_parser.ArgParser(len(sys.argv))
    args = arg_parser.parse()
    train_subjects = re.split('/', args.train_subjects)
    test_subjects = re.split('/', args.test_subjects)

    print('Subjects to train on: ', train_subjects)
    print('Subjects to test on: ', test_subjects)

    config_dict_module = helpers.load_module(args.config_file)
    config_dict = config_dict_module.config_dict
    if config_dict['val_mode'] == 'no_val':
        assert (config_dict['train_mode'] == 'low_level'), \
                'no_val requires low level train mode'
    config_dict['train_subjects'] = train_subjects
    config_dict['test_subjects'] = test_subjects
    if config_dict['val_mode'] == 'subject':
        val_subjects = re.split('/', args.val_subjects)


    config_dict['job_identifier'] = args.job_identifier
    print('Job identifier: ', args.job_identifier)
    wandb.init(project='pfr', config=config_dict)
    wandb.save('models.py')

    all_subjects_df = pd.read_csv(args.subjects_overview)

    # Run the whole program, from preparing the data to evaluating
    # the model's test performance
    run()
