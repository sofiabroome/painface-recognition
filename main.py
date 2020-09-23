import tensorflow as tf
import pandas as pd
import numpy as np
import wandb
import time
import sys
import os
import re

from data_handler import DataHandler
from test_and_eval import Evaluator
from train import train
import compute_steps
import arg_parser
import helpers
import models


TARGET_NAMES = ['NO_PAIN', 'PAIN']
COLOR = True

# Prettyprint for dataframes with long values (filenames).
pd.set_option('max_colwidth', 800)


def df_val_split(df,
                 val_fraction,
                 batch_size):
    """
    If args.val_mode == 'fraction', split the dataframe with training data into two parts,
    a training set and a held out validation set (the last specified fraction from the df).
    :param df: pd.Dataframe
    :param val_fraction: float
    :param batch_size: int
    :param round_to_batch: int [0 or 1]
    :return: pd.Dataframe, pd.Dataframe
    """
    df = df.loc[df['train'] == 1]
    if round_to_batch:
        ns = len(df)
        ns_rounded = ns - ns % batch_size
        num_val = int(val_fraction * ns_rounded - val_fraction * ns_rounded % batch_size)
        df = df.iloc[:ns_rounded]
        df_val = df.iloc[-num_val:, :]
        df_train = df.iloc[:-num_val, :]

    return df_train, df_val


def read_or_create_subject_dfs(dh, subject_ids):
    """
    Read or create the per-subject dataframes listing
    all the frame paths and corresponding labels and metadata.
    :param dh: DataHandler
    :param subject_ids: list of ints referring to subjects
    :return: {str: pd.Dataframe}
    """
    subject_dfs = {}
    for ind, subject_id in enumerate(subject_ids):
        dataset = all_subjects_df.loc[ind]['dataset']
        path_key = 'pf_rgb_path' if dataset == 'pf' else 'lps_rgb_path'
        subject_csv_path = os.path.join(
            config_dict[path_key], subject_id) + '.csv'
        if os.path.isfile(subject_csv_path):
            sdf = pd.read_csv(subject_csv_path)
        else:
            print('Making a DataFrame for: ', subject_id)
            sdf = dh.subject_to_df(subject_id, dataset, config_dict)
            sdf.to_csv(path_or_buf=subject_csv_path)
        subject_dfs[subject_id] = sdf
    return subject_dfs


def read_or_create_subject_rgb_and_OF_dfs(dh,
                                          subject_ids,
                                          subject_dfs):
    """
    Read or create the per-subject optical flow files listing
    all the frame paths and labels.
    :param dh: DataHandler object
    :param subject_ids: list of ints referring to subjects
    :param subject_dfs: [pd.DataFrame]
    :return: {str: pd.Dataframe}
    """
    subject_rgb_OF_dfs = {}
    for ind, subject_id in enumerate(subject_ids):
        dataset = all_subjects_df.loc[ind]['dataset']
        path_key = 'pf_of_path' if dataset == 'pf' else 'lps_of_path'
        subject_of_csv_path = os.path.join(
            config_dict[path_key], subject_id) + '.csv'
        if os.path.isfile(subject_of_csv_path):
            sdf = pd.read_csv(subject_of_csv_path)
        else:
            print('Making a DataFrame with optical flow for: ', subject_id)
            sdf = dh.save_OF_paths_to_df(subject_id,
                                         subject_dfs[subject_id],
                                         dataset=dataset)
            sdf.to_csv(path_or_buf=subject_of_csv_path)
        subject_rgb_OF_dfs[subject_id] = sdf
    return subject_rgb_OF_dfs


def set_train_val_test_in_df(train_subjects,
                             test_subjects,
                             dfs,
                             val_subjects=None):
    """
    Mark in input dataframe which subjects to use for train, val or test.
    Used when val_mode == 'subject'
    :param train_subjects: [int]
    :param val_subjects: [int]
    :param test_subjects: [int]
    :param dfs: [pd.DataFrame]
    :return: [pd.DataFrame]
    """
    for trh in train_subjects:
        dfs[trh]['train'] = 1

    if config_dict['val_mode'] == 'subject':
        for vh in val_subjects:
            dfs[vh]['train'] = 2

    for teh in test_subjects:
        dfs[teh]['train'] = 0
    return dfs


def get_data_indices(dh):
    subject_ids = all_subjects_df['subject'].values

    # Read the dataframes listing all the frame paths and labels
    subject_dfs = read_or_create_subject_dfs(
        dh, subject_ids=subject_ids)

    # If we need optical flow
    if '2stream' in config_dict['model'] or config_dict['data_type'] == 'of':
        subject_dfs = read_or_create_subject_rgb_and_OF_dfs(
            dh=dh,
            subject_ids=subject_ids,
            subject_dfs=subject_dfs)

    # Set the train-column to 1 (train), 2 (val) or 0 (test).
    if config_dict['val_mode'] == 'subject':
        print("Using separate subject validation.")
        val_subjects = re.split('/', args.val_subjects)
        print('Horses to validate on: ', val_subjects)
        subject_dfs = set_train_val_test_in_df(train_subjects=train_subjects,
                                               val_subjects=val_subjects,
                                               test_subjects=test_subjects,
                                               dfs=subject_dfs)

    if config_dict['val_mode'] == 'fraction' or config_dict['val_mode'] == 'no_val':
        subject_dfs = set_train_val_test_in_df(train_subjects=train_subjects,
                                               test_subjects=test_subjects,
                                               dfs=subject_dfs)

    # Put all the separate subject-dfs into one DataFrame.
    df = pd.concat(list(subject_dfs.values()), sort=False)

    print("Total length of dataframe:", len(df))

    # Split training data so there is a held out validation set.
    if config_dict['val_mode'] == 'fraction':
        print("Val fract: ", config_dict['val_fraction_value'])
        df_train, df_val = df_val_split(df=df,
                                        val_fraction=config_dict['val_fraction_value'],
                                        batch_size=config_dict['batch_size'])
    else:
        df_train = df.loc[df['train'] == 1]

    if config_dict['val_mode'] == 'subject':
        df_val = df[df['train'] == 2]

    df_test = df[df['train'] == 0]

    # Reset all indices so they're 0->N.
    print('\nResetting dataframe indices...')
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    if not config_dict['val_mode'] == 'no_val':
        df_val.reset_index(drop=True, inplace=True)
    else:
        df_val = []

    print("Nb. of train, val and test samples: ",
          len(df_train), len(df_val), len(df_test), '\n')

    return df_train, df_val, df_test


def get_data_generators(dh, df_train, df_val, df_test):
    print('\nPreparing data generators...')
    train_gen = dh.get_generator(df_train, train=True)
    val_gen = dh.get_generator(df_val, train=False)
    test_gen = dh.get_generator(df_test, train=False)
    eval_gen = dh.get_generator(df_test, train=False)

    return train_gen, val_gen, test_gen, eval_gen


def get_nb_steps(df, train_str='train'):
    start = time.time()
    train_mode = True if train_str == 'train' else False
    nb_steps, y_batches, y_batches_paths = compute_steps.compute_steps(
        df, train=train_mode, config_dict=config_dict)
    end = time.time()
    print('\nTook {:.2f} s to compute {} {} steps'.format(
        end - start, nb_steps, train_str))

    return nb_steps, y_batches, y_batches_paths


def run():

    print('Batch size: ', config_dict['batch_size'])
    print('Sequence length: ', config_dict['seq_length'])
    dh = DataHandler(data_columns=['pain'],  # Here one can append f. ex. 'Observer',
                     config_dict=config_dict,
                     color=COLOR)

    df_train, df_val, df_test = get_data_indices(dh)

    train_generator,\
        val_generator,\
        test_generator,\
        eval_generator = get_data_generators(dh,
                                             df_train=df_train,
                                             df_val=df_val,
                                             df_test=df_test)

    train_steps, _, _ = get_nb_steps(df_train, 'train')

    test_steps,\
        y_batches,\
        y_batches_paths = get_nb_steps(df_test, 'test')

    if config_dict['val_mode'] == 'no_val':
        val_steps = 0
    else:
        val_steps, _, _ = get_nb_steps(df_val, 'val')

    if args.test_run == 1:
        config_dict['nb_epochs'] = 1
        train_steps = 2
        val_steps = 2
        test_steps = 2
        y_batches = y_batches[:test_steps]
        y_batches_paths = y_batches_paths[:test_steps]

    # Train the model

    model = models.MyModel(config_dict=config_dict)
    best_model_path = train(model_instance=model,
                            config_dict=config_dict,
                            train_steps=train_steps,
                            val_steps=val_steps,
                            generator=train_generator,
                            val_generator=val_generator)

    if config_dict['do_evaluate']:

        model.model.load_weights(best_model_path)
        model = model.model

        ev = Evaluator(acc=True,
                       cm=True,
                       cr=True,
                       auc=True,
                       target_names=TARGET_NAMES,
                       batch_size=config_dict['batch_size'])

        y_preds, scores = ev.test(model=model,
                                  test_generator=test_generator,
                                  eval_generator=eval_generator,
                                  nb_steps=test_steps)

        y_test = np.array(y_batches)  # [nb_batches, batch_size, seq_length, nb_classes]

        if config_dict['nb_input_dims'] == 5:
            # Get the ground truth for the test set
            y_test_paths = np.array(y_batches_paths)
            if args.test_run == 1:
                nb_batches = int(y_preds.shape[0]/config_dict['batch_size'])
                y_test_paths = helpers.flatten_batch_lists(
                    y_batches_paths, nb_batches)
                nb_total = nb_batches * config_dict['batch_size'] * config_dict['seq_length']
                y_test = y_test[:nb_total]
                y_test = np.reshape(y_test, (nb_batches*config_dict['batch_size'],
                                             config_dict['seq_length'],
                                             config_dict['nb_labels']))
            else:
                nb_batches = y_test.shape[0]
                # Make 3D
                y_test = np.reshape(y_test, (nb_batches*config_dict['batch_size'],
                                             config_dict['seq_length'],
                                             config_dict['nb_labels']))
                y_test_paths = np.reshape(y_test_paths, (nb_batches*config_dict['batch_size'],
                                                         config_dict['seq_length']))

        if config_dict['nb_input_dims'] == 4:
            nb_batches = y_test.shape[0]
            y_test = np.reshape(y_test, (nb_batches*config_dict['batch_size'], config_dict['nb_labels']))
            y_test_paths = np.array(y_batches_paths)

        # Put y_preds into same format as y_test, first take the max probabilities.
        if config_dict['nb_input_dims'] == 5:
            if config_dict['rgb_period'] > 1:
                y_preds_argmax = y_preds  # We only have one label per sample, Simonyan case.
                y_test = np.argmax(y_test, axis=1)
            else:
                y_preds_argmax = np.argmax(y_preds, axis=2)
                y_preds_argmax = np.array([tf.keras.utils.to_categorical(x,
                                           num_classes=config_dict['nb_labels']) for x in y_preds_argmax])

        if config_dict['nb_input_dims'] == 4:
            y_preds_argmax = np.argmax(y_preds, axis=1)
            y_preds_argmax = np.array([tf.keras.utils.to_categorical(x,
                                       num_classes=config_dict['nb_labels']) for x in y_preds_argmax])
            if args.test_run == 1:
                y_test = y_test[:len(y_preds)]

        # Evaluate the model's performance
        ev.set_test_set(df_test)
        ev.evaluate(model=model, y_test=y_test, y_pred=y_preds_argmax,
                    softmax_predictions=y_preds, scores=scores,
                    config_dict=config_dict, y_paths=y_test_paths)

if __name__ == '__main__':

    # Parse the command line arguments
    arg_parser = arg_parser.ArgParser(len(sys.argv))
    args = arg_parser.parse()
    train_subjects = re.split('/', args.train_subjects)
    test_subjects = re.split('/', args.test_subjects)

    print('Subjects to train on: ', train_subjects)
    print('Subjects to test on: ', test_subjects)

    config_dict_module = helpers.load_module(args.config_file)
    config_dict = config_dict_module.config_dict
    config_dict['train_subjects'] = args.train_subjects
    config_dict['val_subjects'] = args.val_subjects
    config_dict['test_subjects'] = args.test_subjects
    wandb.init(project='pfr', config=config_dict)

    if config_dict['round_to_batch'] == 1:
        round_to_batch = True
    else:
        round_to_batch = False

    config_dict['job_identifier'] = args.job_identifier
    print('Job identifier: ', args.job_identifier)

    all_subjects_df = pd.read_csv(args.subjects_overview)

    # Run the whole program, from preparing the data to evaluating
    # the model's test performance
    run()
