import pandas as pd
import numpy as np
import keras
import time
import sys
import ast
import os

from data_handler import DataHandler, shuffle_blocks
from test_and_eval import Evaluator
from keras.utils import np_utils
from train import train
import compute_steps
import arg_parser
import helpers
import models

TARGET_NAMES = ['NO_PAIN', 'PAIN']
VAL_FRACTION = 0.2
COLOR = True

# Prettyprint for dataframes with long values (filenames).
pd.set_option('max_colwidth', 800)


def df_val_split(df,
                 val_fraction,
                 batch_size):
    """
    If args.val_fraction == 1, split the dataframe with training data into two parts,
    a training set and a held out validation set (the last specified fraction from the df).
    :param df: pd.Dataframe
    :param val_fraction: float
    :param batch_size: int
    :param round_to_batch: int [0 or 1]
    :return: pd.Dataframe, pd.Dataframe
    """
    df = df.loc[df['Train'] == 1]
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
            print('Making a DataFrame for subject id: ', subject_id)
            sdf = dh.subject_to_df(subject_id, dataset, config_dict)
            sdf.to_csv(path_or_buf=subject_csv_path)
        subject_dfs[ind] = sdf
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
        subject_of_csv_path = os.path.join(
            args.data_path, args.of_path, subject_id) + '.csv'
        if os.path.isfile(subject_of_csv_path):
            sdf = pd.read_csv(subject_of_csv_path)
        else:
            print('Making a DataFrame for subject id: ', subject_id)
            sdf = dh.save_OF_paths_to_df(subject_id,
                                         subject_dfs[ind])
            sdf.to_csv(path_or_buf=subject_of_csv_path)
        subject_rgb_OF_dfs[ind] = sdf
    return subject_rgb_OF_dfs


def set_train_val_test_in_df(train_subjects,
                             test_subjects,
                             dfs,
                             val_subjects=None):
    """
    Mark in input dataframe which subjects to use for train, val or test.
    Used when args.val_fraction == 0.
    :param train_subjects: [int]
    :param val_subjects: [int]
    :param test_subjects: [int]
    :param dfs: [pd.DataFrame]
    :return: [pd.DataFrame]
    """
    for trh in train_subjects:
        dfs[trh]['Train'] = 1

    for vh in val_subjects:
        dfs[vh]['Train'] = 2

    for teh in test_subjects:
        dfs[teh]['Train'] = 0
    return dfs


def run():

    print('Batch size: ', config_dict['batch_size'])
    print('Sequence length: ', config_dict['seq_length'])

    subject_ids = all_subjects_df['subject'].values

    train_subjects = ast.literal_eval(args.train_subjects)
    test_subjects = ast.literal_eval(args.test_subjects)

    print('Subjects to train on: ', train_subjects)
    print('Subjects to test on: ', test_subjects)

    model = models.MyModel(config_dict=config_dict)

    dh = DataHandler(path=args.rgb_path,
                     of_path=args.of_path,
                     data_columns=['pain'],  # Here one can append f. ex. 'Observer',
                     config_dict=config_dict,
                     color=COLOR)
    ev = Evaluator(acc=True,
                   cm=True,
                   cr=True,
                   auc=True,
                   target_names=TARGET_NAMES,
                   batch_size=config_dict['batch_size'])

    # Read or create the per-subject dataframes listing all the frame paths and labels.
    subject_dfs = read_or_create_subject_dfs(
        dh, subject_ids=subject_ids)  # Returns a dict of dataframes, per subject.

    # If we need optical flow
    if '2stream' in config_dict['model'] or config_dict['data_type'] == 'of':
        subject_dfs = read_or_create_subject_rgb_and_OF_dfs(
            dh=dh,
            subject_ids=subject_ids,
            subject_dfs=subject_dfs)

    # Set the train-column to 1 (train), 2 (val) or 0 (test).
    if config_dict['val_fraction'] == 0:
        print("Using separate subject validation.")
        val_subjects = ast.literal_eval(args.val_subjects)
        print('Horses to validate on: ', val_subjects)
        subject_dfs = set_train_val_test_in_df(train_subjects=train_subjects,
                                               val_subjects=val_subjects,
                                               test_subjects=test_subjects,
                                               dfs=subject_dfs)

    if config_dict['val_fraction'] == 1:
        print("Val fract: ", VAL_FRACTION)
        subject_dfs = set_train_val_test_in_df(train_subjects=train_subjects,
                                               test_subjects=test_subjects,
                                               dfs=subject_dfs)

    # Put all the separate subject-dfs into one DataFrame.
    df = pd.concat(list(subject_dfs.values()))

    print("Total length of dataframe:", len(df))

    # Split training data so there is a held out validation set.
    if config_dict['val_fraction'] == 1:
        df_train, df_val = df_val_split(df=df,
                                        val_fraction=VAL_FRACTION,
                                        batch_size=config_dict['batch_size'])
    if config_dict['val_fraction'] == 0:
        df_train = df[df['Train'] == 1]
        df_val = df[df['Train'] == 2]

    df_test = df[df['Train'] == 0]

    # Count the number of samples in each partition of the data.
    nb_train_samples = len(df_train)
    nb_val_samples = len(df_val)
    nb_test_samples = len(df_test)

    print("Lengths dftr, dfval and dftest: ", nb_train_samples, nb_val_samples, nb_test_samples)

    # Reset all indices so they're 0->N.
    print('Resetting dataframe indices...')
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    train_generator = dh.get_generator(df_train, train=True)
    val_generator = dh.get_generator(df_val, train=False)
    test_generator = dh.get_generator(df_test, train=False)
    eval_generator = dh.get_generator(df_test, train=False)

    start = time.time()
    train_steps, _, _ = compute_steps.compute_steps(
        df_train, train=True, config_dict=config_dict)
    end = time.time()
    print('Took {} s to compute training steps'.format(end - start))

    start = time.time()
    val_steps, _, _ = compute_steps.compute_steps(
        df_val, train=False, config_dict=config_dict)
    end = time.time()
    print('Took {} s to compute validation steps'.format(end - start))

    start = time.time()
    test_steps, y_batches, y_batches_paths = compute_steps.compute_steps(
        df_test, train=False, config_dict=config_dict)
    end = time.time()
    print('Took {} s to compute testing steps'.format(end - start))

    if args.test_run == 1:
        config_dict['nb_epochs'] = 1
        train_steps = 2
        val_steps = 2
        test_steps = 2
        y_batches = y_batches[:test_steps]
        y_batches_paths = y_batches_paths[:test_steps]

    # Train the model
    best_model_path = train(model_instance=model,
                            config_dict=config_dict,
                            train_steps=train_steps,
                            val_steps=val_steps,
                            val_fraction=VAL_FRACTION,
                            generator=train_generator,
                            val_generator=val_generator)

    model = keras.models.load_model(best_model_path)

    # Get test predictions
    y_preds, scores = ev.test(model=model,
                              test_generator=test_generator,
                              eval_generator=eval_generator,
                              nb_steps=test_steps)

    y_test = np.array(y_batches)  # Now in format [nb_batches, batch_size, seq_length, nb_classes]

    if config_dict['nb_input_dims'] == 5:
        # Get the ground truth for the test set
        y_test_paths = np.array(y_batches_paths)
        if args.test_run == 1:
            y_test_paths = y_batches_paths[:2]
            nb_batches = int(y_preds.shape[0]/config_dict['batch_size'])
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
            y_preds_argmax = np.array([np_utils.to_categorical(x,
                                       num_classes=config_dict['nb_labels']) for x in y_preds_argmax])
    
    if config_dict['nb_input_dims'] == 4:
        y_preds_argmax = np.argmax(y_preds, axis=1)
        y_preds_argmax = np.array([np_utils.to_categorical(x,
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
    config_dict_module = helpers.load_module(args.config_file)
    config_dict = config_dict_module.config_dict

    if config_dict['round_to_batch'] == 1:
        round_to_batch = True
    else:
        round_to_batch = False

    config_dict['job_identifier'] = args.job_identifier

    all_subjects_df = pd.read_csv(args.subjects_overview)

    # Run the whole program, from preparing the data to evaluating
    # the model's test performance
    run()
