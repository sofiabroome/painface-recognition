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


def read_or_create_horse_dfs(dh):
    """
    Read or create the per-horse dataframes listing
    all the frame paths and corresponding labels and metadata.
    :param dh: DataHandler
    :return: [pd.Dataframe]
    """
    horse_dfs = []
    for horse in range(1, 7):
        print(args.data_path)
        horse_csv_path = args.data_path + 'horse_' + str(horse) + '.csv'
        if os.path.isfile(horse_csv_path):
            hdf = pd.read_csv(horse_csv_path)
        else:
            print('Making a DataFrame for horse id: ', horse)
            hdf = dh.horse_to_df(horse)
            hdf.to_csv(path_or_buf=horse_csv_path)
        horse_dfs.append(hdf)
    return horse_dfs


def read_or_create_horse_rgb_and_OF_dfs(dh,
                                        horse_dfs):
    """
    Read or create the per-horse optical flow files listing
    all the frame paths and labels.
    :param dh: DataHandler object
    :param horse_dfs: [pd.DataFrame]
    :return: [pd.DataFrame]
    """
    horse_rgb_OF_dfs = []
    for horse_id in range(1, 7):
        print(args.data_path)
        horse_of_csv_path = dh.of_path + '/horse_' + str(horse_id) + '.csv'
        if os.path.isfile(horse_of_csv_path):
            hdf = pd.read_csv(horse_of_csv_path)
        else:
            print('Making a DataFrame for horse id: ', horse_id)
            hdf = dh.save_OF_paths_to_df(horse_id,
                                         horse_dfs[horse_id-1])
            hdf.to_csv(path_or_buf=horse_of_csv_path)
        horse_rgb_OF_dfs.append(hdf)
    return horse_rgb_OF_dfs


def set_train_val_test_in_df(train_horses,
                             val_horses,
                             test_horses,
                             dfs):
    """
    Mark in input dataframe which horses to use for train, val or test.
    Used when args.val_fraction == 0.
    :param train_horses: [int]
    :param val_horses: [int]
    :param test_horses: [int]
    :param dfs: [pd.DataFrame]
    :return: [pd.DataFrame]
    """
    for trh in train_horses:
        dfs[trh]['Train'] = 1

    for vh in val_horses:
        dfs[vh]['Train'] = 2

    for teh in test_horses:
        dfs[teh]['Train'] = 0
    return dfs


def set_train_test_in_df(train_horses,
                         test_horses,
                         dfs):
    """
    Mark in input dataframe which horses to use for train or test.
    Used when args.val_fraction == 1.
    :param train_horses: [int]
    :param test_horses: [int]
    :param dfs: [pd.DataFrame]
    :return: [pd.DataFrame]
    """
    for trh in train_horses:
        dfs[trh]['Train'] = 1

    for teh in test_horses:
        dfs[teh]['Train'] = 0
    return dfs


def get_data_4d_input(dh,
                      data_type,
                      df_train,
                      df_test,
                      df_val=None):
    """
    Prepare the training and testing data for 4D-input (batches of frames)
    :param dh: DataHandler object
    :param data_type: str ['rgb' || 'of']
    :param df_train: pd.DataFrame
    :param df_test: pd.DataFrame
    :param df_val: pd.DataFrame
    :return: (4-tuple of Generator objects)
    """
    train_generator = dh.prepare_image_generator(df_train, data_type,
                                                 train=True, val=False,
                                                 test=False, evaluate=False)
    val_generator = dh.prepare_image_generator(df_val, data_type,
                                               train=False, val=True,
                                               test=False, evaluate=False)
    test_generator = dh.prepare_image_generator(df_test, data_type,
                                                train=False, val=False,
                                                test=True, evaluate=False)
    eval_generator = dh.prepare_image_generator(df_test, data_type,
                                                train=False, val=False,
                                                test=False,  evaluate=True)
    generators = (train_generator, val_generator, test_generator, eval_generator)
    return generators


def get_data_5d_input(dh,
                      data_type,
                      df_train,
                      df_val,
                      df_test):
    """
    Prepare the training and testing data for 5D-input
    (batches of sequences of frames).
    :param dh: DataHandler object
    :param data_type: str ['rgb' || 'of']
    :param df_train: pd.DataFrame
    :param df_val: pd.DataFrame
    :param df_test: pd.DataFrame
    :return: (4-tuple of Generator objects)
    """
    train_generator = dh.prepare_image_generator_5D(df_train,
                                                    data_type=data_type,
                                                    train=True, val=False,
                                                    test=False, evaluate=False)
    val_generator = dh.prepare_image_generator_5D(df_val,
                                                  data_type=data_type,
                                                  train=False, val=True,
                                                  test=False, evaluate=False)
    test_generator = dh.prepare_image_generator_5D(df_test,
                                                   data_type=data_type,
                                                   train=False, val=False,
                                                   test=True, evaluate=False)
    eval_generator = dh.prepare_image_generator_5D(df_test,
                                                   data_type=data_type,
                                                   train=False, val=False,
                                                   test=False, evaluate=True)
    generators = (train_generator, val_generator, test_generator, eval_generator)
    return generators


def get_data_2stream_4d_input(dh,
                              df_train_rgbof,
                              df_val_rgbof,
                              df_test_rgbof):
    """
    Prepare data generators for the 2stream model with 4D input.
    :param dh: DataHandler object
    :param df_train_rgbof: pd.DataFrame
    :param df_val_rgbof: pd.DataFrame
    :param df_test_rgbof: pd.DataFrame
    :return: (4-tuple of Generator objects)
    """
    train_generator = dh.prepare_generator_2stream(df_train_rgbof,
                                                   train=True, val=False,
                                                   test=False, evaluate=False)
    val_generator = dh.prepare_generator_2stream(df_val_rgbof,
                                                 train=False, val=True,
                                                 test=False, evaluate=False)
    test_generator = dh.prepare_generator_2stream(df_test_rgbof,
                                                  train=False, val=False,
                                                  test=True, evaluate=False)
    eval_generator = dh.prepare_generator_2stream(df_test_rgbof,
                                                  train=False, val=False,
                                                  test=False, evaluate=True)
    generators = (train_generator, val_generator, test_generator, eval_generator)
    return generators


def get_data_2stream_5d_input(dh,
                              df_train_rgbof,
                              df_val_rgbof,
                              df_test_rgbof):
    """
    Prepare the training and testing data for 5D-input
    (batches of sequences of frames).
    :param dh: DataHandler object
    :param horse_dfs: [pd.DataFrame]
    :param train_horses: [int]
    :param test_horses: [int]
    :param val_horses: [int]
    :return: (4-tuple of Generator objects)
    """
    print("2stream model of some sort.", args.model)

    print("Using the 5D generator for 2stream")
    train_generator = dh.prepare_2stream_image_generator_5D(df_train_rgbof,
                                                            train=True,
                                                            val=False,
                                                            test=False,
                                                            evaluate=False)
    val_generator = dh.prepare_2stream_image_generator_5D(df_val_rgbof,
                                                          train=False,
                                                          val=True,
                                                          test=False,
                                                          evaluate=False)
    test_generator = dh.prepare_2stream_image_generator_5D(df_test_rgbof,
                                                           train=False,
                                                           val=False,
                                                           test=True,
                                                           evaluate=False)
    eval_generator = dh.prepare_2stream_image_generator_5D(df_test_rgbof,
                                                           train=False,
                                                           val=False,
                                                           test=False,
                                                           evaluate=True)
    generators = (train_generator, val_generator, test_generator, eval_generator)
    return generators


def run():

    # Some initial print outs to keep track of the training mode,
    # what horses and batch size etc.

    print('Batch size:')
    print(args.batch_size)
    print('Sequence length:')
    print(args.seq_length)

    train_horses = ast.literal_eval(args.train_horses)
    test_horses = ast.literal_eval(args.test_horses)

    print('Horses to train on: ', train_horses)
    print('Horses to test on: ', test_horses)

    model = models.MyModel(args=args)
    dh = DataHandler(path=args.data_path,
                     of_path=args.of_path,
                     image_size=(args.input_width, args.input_height),
                     seq_length=args.seq_length,
                     seq_stride=args.seq_stride,
                     batch_size=args.batch_size,
                     color=COLOR,
                     nb_labels=args.nb_labels,
                     aug_flip=args.aug_flip,
                     aug_crop=args.aug_crop,
                     aug_light=args.aug_light)

    ev = Evaluator(acc=True,
                   cm=True,
                   cr=True,
                   target_names=TARGET_NAMES,
                   batch_size=args.batch_size)

    # Read or create the per-horse dataframes listing all the frame paths and labels.
    horse_dfs = read_or_create_horse_dfs(dh)  # Returns a list of dataframes, per horse.

    if '2stream' in args.model or args.data_type == 'of':
        horse_dfs = read_or_create_horse_rgb_and_OF_dfs(dh=dh,
                                                        horse_dfs=horse_dfs)
    # Set the train-column to 1 (train), 2 (val) or 0 (test).
    if args.val_fraction == 0:
        print("Using separate horse validation.")
        val_horses = ast.literal_eval(args.val_horses)
        print('Horses to validate on: ', val_horses)
        horse_dfs = set_train_val_test_in_df(train_horses=train_horses,
                                             val_horses=val_horses,
                                             test_horses=test_horses,
                                             dfs=horse_dfs)

    if args.val_fraction == 1:
        print("Using validation fraction.")
        print("Val fract: ", VAL_FRACTION)
        horse_dfs = set_train_test_in_df(train_horses=train_horses,
                                         test_horses=test_horses,
                                         dfs=horse_dfs)

    # Put all the separate horse-dfs into one DataFrame.
    df = pd.concat(horse_dfs)

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
            generators = get_data_2stream_5d_input(dh=dh,
                                                   df_train_rgbof=df_train,
                                                   df_val_rgbof=df_val,
                                                   df_test_rgbof=df_test)
        else:
            print('5d input model')
            if args.data_type == 'rgb':
                print('Only RGB data')
            if args.data_type == 'of':
                print('Only optical flow data')
            generators = get_data_5d_input(dh=dh,
                                           data_type=args.data_type,
                                           df_train=df_train,
                                           df_val=df_val,
                                           df_test=df_test)
    if args.nb_input_dims == 4:
        if '2stream' in args.model:
            print('4d input 2stream model. Needs (quick) fix')
            #  generators = get_data_2stream_4d_input(dh=dh,
            #                                         horse_dfs,
            #                                         train_horses,
            #                                         test_horses,
            #                                         val_horses)
        else:
            print('4d input model')
            if args.data_type == 'rgb':
                print('Only RGB data')
            if args.data_type == 'of':
                print('Only optical flow data')
            generators = get_data_4d_input(dh=dh,
                                           data_type=args.data_type,
                                           df_train=df_train,
                                           df_test=df_test,
                                           df_val=df_val)

    train_generator, val_generator, test_generator, eval_generator = generators

    if args.test_run == 1:
        train_steps = 2
        val_steps = 2
        test_steps = 2
        y_test = df_test['Pain'].values
    else:
        start = time.time()
        train_steps, _ = compute_steps.compute_steps(df_train, args)
        end = time.time()
        print('Took {} s to compute training steps'.format(end - start))

        start = time.time()
        val_steps, _ = compute_steps.compute_steps(df_val, args)
        end = time.time()
        print('Took {} s to compute validation steps'.format(end - start))

        start = time.time()
        test_steps, y_batches = compute_steps.compute_steps(df_test, args)
        end = time.time()
        print('Took {} s to compute testing steps'.format(end - start))

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

    if args.test_run == 1:
        nb_batches = y_preds.shape[0]
        nb_total = nb_batches * args.batch_size * args.seq_length
        y_test = np_utils.to_categorical(y_test)
        y_test = y_test[:nb_total]
        y_test = np.reshape(y_test, (nb_batches*args.batch_size, args.seq_length, args.nb_labels))
        y_test = y_test[:nb_batches]
    else:
        # Get the ground truth for the test set
        y_test = np.array(y_batches)  # Now in format [nb_batches, batch_size, seq_length, nb_classes]
        nb_batches = y_test.shape[0]
        # Make 3D
        y_test = np.reshape(y_test, (nb_batches*args.batch_size, args.seq_length, args.nb_labels))

    # Put y_preds into same format as y_test, first take the max probabilities.
    y_preds = np.argmax(y_preds, axis=2)
    y_preds = np.array([np_utils.to_categorical(x, num_classes=args.nb_labels) for x in y_preds])

    # Evaluate the model's performance
    ev.evaluate(model=model, y_test=y_test, y_pred=y_preds, scores=scores, args=args)

if __name__ == '__main__':

    # Parse the command line arguments
    arg_parser = arg_parser.ArgParser(len(sys.argv))
    args = arg_parser.parse()

    if args.round_to_batch == 1:
        round_to_batch = True
    else:
        round_to_batch = False

    # Run the whole program, from preparing the data to evaluating
    # the model's test performance
    run()
