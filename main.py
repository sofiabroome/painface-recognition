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
    :return: [pd.Dataframe]
    """
    subject_dfs = []
    for subject_id in subject_ids:
        print(args.data_path)
        subject_csv_path = args.data_path + subject_id + '.csv'
        if os.path.isfile(subject_csv_path):
            sdf = pd.read_csv(subject_csv_path)
        else:
            print('Making a DataFrame for subject id: ', subject_id)
            sdf = dh.subject_to_df(subject_id)
            sdf.to_csv(path_or_buf=subject_csv_path)
        subject_dfs.append(sdf)
    return subject_dfs


def read_or_create_subject_rgb_and_OF_dfs(dh,
                                          subject_ids,
                                          subject_dfs):
    """
    Read or create the per-subject optical flow files listing
    all the frame paths and labels.
    :param dh: DataHandler object
    :param subject_dfs: [pd.DataFrame]
    :return: [pd.DataFrame]
    """
    subject_rgb_OF_dfs = []
    for ind, subject_id in enumerate(subject_ids):
        subject_of_csv_path = dh.of_path + str(subject_id) + '.csv'
        if os.path.isfile(subject_of_csv_path):
            sdf = pd.read_csv(subject_of_csv_path)
        else:
            print('Making a DataFrame for subject id: ', subject_id)
            sdf = dh.save_OF_paths_to_df(subject_id,
                                         subject_dfs[ind])
            sdf.to_csv(path_or_buf=subject_of_csv_path)
        subject_rgb_OF_dfs.append(sdf)
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


def get_data_generators(prepare_generator_func,
                        df_train,
                        df_test,
                        df_val=None):
    """
    Prepare the training and testing data for 4D-input (batches of frames)
    :param prepare_generator_func: function to call to get generator
    :param df_train: pd.DataFrame
    :param df_test: pd.DataFrame
    :param df_val: pd.DataFrame
    :return: (4-tuple of Generator objects)
    """
    train_generator = prepare_generator_func(df_train,
                                             train=True, config_dict=config_dict)
    val_generator = prepare_generator_func(df_val,
                                           train=False, config_dict=config_dict)
    test_generator = prepare_generator_func(df_test,
                                            train=False, config_dict=config_dict)
    eval_generator = prepare_generator_func(df_test,
                                            train=False, config_dict=config_dict)
    return train_generator, val_generator, test_generator, eval_generator


def run():

    # Some initial print outs to keep track of the training mode,
    # what subjects and batch size etc.

    print('Batch size:')
    print(args.batch_size)
    print('Sequence length:')
    print(args.seq_length)

    subject_ids = pd.read_csv(args.subjects_overview)['Subject'].values

    train_subjects = ast.literal_eval(args.train_subjects)
    test_subjects = ast.literal_eval(args.test_subjects)

    print('Subjects to train on: ', train_subjects)
    print('Subjects to test on: ', test_subjects)

    model = models.MyModel(args=args)
    dh = DataHandler(path=args.data_path,
                     of_path=args.of_path,
                     clip_list_file=args.data_path + 'overview.csv',
                     data_columns=['Pain'],  # Here one can append f. ex. 'Observer'
                     image_size=(args.input_width, args.input_height),
                     seq_length=args.seq_length,
                     seq_stride=args.seq_stride,
                     batch_size=args.batch_size,
                     color=COLOR,
                     nb_labels=args.nb_labels,
                     aug_flip=args.aug_flip,
                     aug_crop=args.aug_crop,
                     aug_light=args.aug_light,
                     nb_input_dims=args.nb_input_dims)

    ev = Evaluator(acc=True,
                   cm=True,
                   cr=True,
                   auc=True,
                   target_names=TARGET_NAMES,
                   batch_size=args.batch_size)

    # Read or create the per-subject dataframes listing all the frame paths and labels.
    subject_dfs = read_or_create_subject_dfs(dh, subject_ids)  # Returns a list of dataframes, per subject.

    if '2stream' in args.model or args.data_type == 'of':
        subject_dfs = read_or_create_subject_rgb_and_OF_dfs(dh=dh,
                                                            subject_ids=subject_ids,
                                                            subject_dfs=subject_dfs)
    # Set the train-column to 1 (train), 2 (val) or 0 (test).
    if args.val_fraction == 0:
        print("Using separate subject validation.")
        val_subjects = ast.literal_eval(args.val_subjects)
        print('Horses to validate on: ', val_subjects)
        subject_dfs = set_train_val_test_in_df(train_subjects=train_subjects,
                                               val_subjects=val_subjects,
                                               test_subjects=test_subjects,
                                               dfs=subject_dfs)

    if args.val_fraction == 1:
        print("Using validation fraction.")
        print("Val fract: ", VAL_FRACTION)
        subject_dfs = set_train_val_test_in_df(train_subjects=train_subjects,
                                               test_subjects=test_subjects,
                                               dfs=subject_dfs)

    # Put all the separate subject-dfs into one DataFrame.
    df = pd.concat(subject_dfs)

    # Shuffle the different sequences (like 1_1a_1) so that they don't always
    # appear in the same order. Also done in training generator but we do it here so that
    # the validation set is more random as well.
    df = shuffle_blocks(df, 'Video_ID') # The df-index is now shuffled as well, if that matters.
    print("Total length of dataframe:", len(df))

    # Split training data so there is a held out validation set.
    if args.val_fraction == 1:
        df_train, df_val = df_val_split(df=df,
                                        val_fraction=VAL_FRACTION,
                                        batch_size=args.batch_size)
    if args.val_fraction == 0:
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

    # Prepare the training and testing data, format depends on model.
    # (5D/4D -- 2stream/1stream)

    if args.nb_input_dims == 5:

        if '2stream' in args.model:
            print('5d input 2stream model')
                
            generators = get_data_generators(
                prepare_generator_func=dh.prepare_2stream_image_generator_5D(),
                df_train_rgbof=df_train,
                df_val_rgbof=df_val,
                df_test_rgbof=df_test)
        else:
            print('5d input model')
            if config_dict['data_type'] == 'rgb':
                print('Only RGB data')
            if config_dict['data_type'] == 'of':
                print('Only optical flow data')
            generators = get_data_generators(
                prepare_generator_func=dh.prepare_image_generator_5D(),
                df_train=df_train,
                df_val=df_val,
                df_test=df_test)
    if args.nb_input_dims == 4:
        if '2stream' in args.model:
            print('4d input 2stream model. Needs (quick) fix')
            generators = get_data_generators(
                prepare_generator_func=dh.prepare_generator_2stream(),
                df_train_rgbof=df_train,
                df_val_rgbof=df_val,
                df_test_rgbof=df_test)
        else:
            print('4d input model')
            if args.data_type == 'rgb':
                print('Only RGB data')
            if args.data_type == 'of':
                print('Only optical flow data')
            generators = get_data_generators(
                prepare_generator_func=dh.prepare_image_generator(),
                df_train=df_train,
                df_test=df_test,
                df_val=df_val)

    train_generator, val_generator, test_generator, eval_generator = generators
    
    start = time.time()
    train_steps, _, _ = compute_steps.compute_steps(df_train, train=True, kwargs=args)
    end = time.time()
    print('Took {} s to compute training steps'.format(end - start))

    start = time.time()
    val_steps, _, _ = compute_steps.compute_steps(df_val, train=False, kwargs=args)
    end = time.time()
    print('Took {} s to compute validation steps'.format(end - start))

    start = time.time()
    test_steps, y_batches, y_batches_paths = compute_steps.compute_steps(df_test, train=False, kwargs=args)
    end = time.time()
    print('Took {} s to compute testing steps'.format(end - start))

    if args.test_run == 1:
        train_steps = 2
        val_steps = 2
        test_steps = 2
        y_batches = y_batches[:test_steps]
        y_batches_paths = y_batches_paths[:test_steps]

    # Train the model
    best_model_path = train(model_instance=model,
                            args=args,
                            train_steps=train_steps,
                            val_steps=val_steps,
                            val_fraction=VAL_FRACTION,
                            generator=train_generator,
                            val_generator=val_generator)

    model = keras.models.load_model(best_model_path)

    # Get test predictions
    y_preds, scores = ev.test(model=model, args=args,
                              test_generator=test_generator,
                              eval_generator=eval_generator,
                              nb_steps=test_steps)

    y_test = np.array(y_batches)  # Now in format [nb_batches, batch_size, seq_length, nb_classes]

    if args.nb_input_dims == 5:
        # Get the ground truth for the test set
        y_test_paths = np.array(y_batches_paths)
        if args.test_run == 1:
            y_test_paths = y_batches_paths[:2]
            nb_batches = int(y_preds.shape[0]/args.batch_size)
            nb_total = nb_batches * args.batch_size * args.seq_length
            y_test = y_test[:nb_total]
            y_test = np.reshape(y_test, (nb_batches*args.batch_size,
                                         args.seq_length,
                                         args.nb_labels))
        else:
            nb_batches = y_test.shape[0]
            # Make 3D
            y_test = np.reshape(y_test, (nb_batches*args.batch_size,
                                         args.seq_length,
                                         args.nb_labels))
            y_test_paths = np.reshape(y_test_paths, (nb_batches*args.batch_size,
                                                     args.seq_length))

    if args.nb_input_dims == 4:
        nb_batches = y_test.shape[0]
        y_test = np.reshape(y_test, (nb_batches*args.batch_size, args.nb_labels))
        y_test_paths = np.array(y_batches_paths)

    # Put y_preds into same format as y_test, first take the max probabilities.
    if args.nb_input_dims == 5:
        if rgb_period > 1:
            y_preds_argmax = y_preds  # We only have one label per sample, Simonyan case.
            y_test = np.argmax(y_test, axis=1)
        else:
            y_preds_argmax = np.argmax(y_preds, axis=2)
            y_preds_argmax = np.array([np_utils.to_categorical(x,
                                       num_classes=args.nb_labels) for x in y_preds_argmax])
    
    if args.nb_input_dims == 4:
        y_preds_argmax = np.argmax(y_preds, axis=1)
        y_preds_argmax = np.array([np_utils.to_categorical(x,
                                   num_classes=args.nb_labels) for x in y_preds_argmax])
        if args.test_run == 1:
            y_test = y_test[:len(y_preds)]

    # Evaluate the model's performance
    ev.set_test_set(df_test)
    ev.evaluate(model=model, y_test=y_test, y_pred=y_preds_argmax,
                softmax_predictions=y_preds, scores=scores, args=args, y_paths=y_test_paths)

if __name__ == '__main__':

    # Parse the command line arguments
    arg_parser = arg_parser.ArgParser(len(sys.argv))
    args = arg_parser.parse()

    if args.round_to_batch == 1:
        round_to_batch = True
    else:
        round_to_batch = False

    config_dict_module = helpers.load_module(args.config_file)
    config_dict = config_dict_module.config_dict
    config_dict['job_identifier'] = args.job_identifier
    config_dict['data_dir_path'] = args.dataset_path
    config_dict['dataset_folder_train'] = args.dataset_path
    config_dict['dataset_folder_test'] = args.dataset_path

    # Run the whole program, from preparing the data to evaluating
    # the model's test performance
    run()
