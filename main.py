import _pickle as cp
import pandas as pd
import numpy as np
import sys
import ast
import os

from data_handler import DataHandler, shuffle_blocks
from test_and_eval import Evaluator
from train import train
import arg_parser
import models

TARGET_NAMES = ['NO_PAIN', 'PAIN']
VAL_FRACTION = 0.2
COLOR = True

pd.set_option('max_colwidth', 800)
# np.random.seed(100)


def df_val_split(df, val_fraction, batch_size, round_to_batch=True):
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
    # Read or create the per-horse dataframes listing all the frame paths and labels.
    horse_dfs = []
    for horse in range(1,7):
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


def read_or_create_horse_rgb_and_OF_dfs(dh, horse_dfs):
    # Read or create the per-horse optical flow files listing all the frame paths and labels.
    horse_rgb_OF_dfs = []
    for horse_id in range(1, 7):
        print(args.data_path)
        horse_of_csv_path = 'data/jpg_320_180_1fps_OF/horse_' + str(horse_id) + '.csv'
        if os.path.isfile(horse_of_csv_path):
            hdf = pd.read_csv(horse_of_csv_path)
        else:
            print('Making a DataFrame for horse id: ', horse_id)
            hdf = dh.save_OF_paths_to_df(horse_id, horse_dfs[horse_id-1])
            hdf.to_csv(path_or_buf=horse_of_csv_path)
        horse_rgb_OF_dfs.append(hdf)
    return horse_rgb_OF_dfs


def set_train_test_in_df(train_horses, test_horses, dfs):
    for trh in train_horses:
        dfs[trh]['Train'] = 1

    for teh in test_horses:
        dfs[teh]['Train'] = 0
    return dfs


def run():

    # Some initial print outs to keep track of the training mode,
    # what horses and batch size etc.
    print('Batch size:')
    print(args.batch_size)
    print('Sequence length:')
    print(args.seq_length)
    print("(Normally they should be the same, except in 5D input mode.)")

    train_horses = ast.literal_eval(args.train_horses)
    test_horses = ast.literal_eval(args.test_horses)
    print('Horses to train on: ', train_horses)
    print('Horses to test on: ', test_horses)

    model = models.MyModel(args)
    dh = DataHandler(args.data_path, (args.input_width, args.input_height),
                     args.seq_length, args.batch_size, COLOR, args.nb_labels)
    ev = Evaluator(True, True, True, TARGET_NAMES, args.batch_size)

    # Read or create the per-horse dataframes listing all the frame paths and labels.
    horse_dfs = read_or_create_horse_dfs(dh)

    # Set the train-column to 1 (yes) or 0 (no).
    horse_dfs = set_train_test_in_df(train_horses, test_horses, horse_dfs)

    # Put all the separate horse-dfs into one DataFrame.
    df = pd.concat(horse_dfs)

    # Shuffle the different sequences (like 1_1a_1) so that they don't always
    # appear in the same order. Also done in generator but we do it here so that
    # the validation set is more random as well.
    df = shuffle_blocks(df)

    # Split training data so there is a held out validation set.
    print("lengths df:", len(df))
    df_train, df_val = df_val_split(df, val_fraction=VAL_FRACTION,
                                    batch_size=args.batch_size,
                                    round_to_batch=args.round_to_batch)
    print("lengths dftr and df val:", len(df_train), len(df_val))

    # Count the number of samples in each partition of the data.
    nb_train_samples = len(df_train)
    nb_val_samples = len(df_val)
    nb_test_samples = len(df[df['Train'] == 0])

    # Prepare the training and testing data for 5D-input (batches of videos)
    # if 'timedist' in args.model or '5d' in args.model or '3d' in args.model:
    if 'timedist' in args.model or '3d' in args.model:
        print('5d input model')
        train_generator = dh.prepare_image_generator_5D(df_train, train=True, val=False, test=False, eval=False)
        val_generator = dh.prepare_image_generator_5D(df_val, train=False, val=True, test=False, eval=False)
        test_generator = dh.prepare_image_generator_5D(df[df['Train'] == 0], train=False, val=False, test=True, eval=False)
        eval_generator = dh.prepare_image_generator_5D(df[df['Train'] == 0], train=False, val=False, test=False, eval=True)

    elif '2stream' in args.model:
        print("2stream model of some sort.", args.model)
        # Read or create the per-horse optical flow files listing all the frame paths and labels.
        horse_rgb_OF_dfs = read_or_create_horse_rgb_and_OF_dfs(dh, horse_dfs)
        horse_rgb_OF_dfs = set_train_test_in_df(train_horses, test_horses, horse_rgb_OF_dfs)
        df_rgb_and_of = pd.concat(horse_rgb_OF_dfs)
        df_rgb_and_of = shuffle_blocks(df_rgb_and_of)
        # Split into train and test
        df_train_rgbof, df_val_rgbof = df_val_split(df_rgb_and_of,
                                                    val_fraction=VAL_FRACTION,
                                                    batch_size=args.batch_size,
                                                    round_to_batch=args.round_to_batch)
        if '5d' in args.model:
            print("Using the 5D generator for 2stream")
            train_generator = dh.prepare_2stream_image_generator_5D(df_train_rgbof,
                                                                    train=True, val=False, test=False, eval=False)
            val_generator = dh.prepare_2stream_image_generator_5D(df_val_rgbof,
                                                         train=False, val=True, test=False, eval=False)
            test_generator = dh.prepare_2stream_image_generator_5D(df_rgb_and_of[df_rgb_and_of['Train'] == 0],
                                                          train=False, val=False, test=True, eval=False)
            eval_generator = dh.prepare_2stream_image_generator_5D(df_rgb_and_of[df_rgb_and_of['Train'] == 0],
                                                          train=False, val=False, test=False, eval=True)
        else:
            train_generator = dh.prepare_generator_2stream(df_train_rgbof,
                                                           train=True, val=False, test=False, eval=False)
            val_generator = dh.prepare_generator_2stream(df_val_rgbof,
                                                         train=False, val=True, test=False, eval=False)
            test_generator = dh.prepare_generator_2stream(df_rgb_and_of[df_rgb_and_of['Train'] == 0],
                                                          train=False, val=False, test=True, eval=False)
            eval_generator = dh.prepare_generator_2stream(df_rgb_and_of[df_rgb_and_of['Train'] == 0],
                                                          train=False, val=False, test=False, eval=True)
    # Prepare the training and testing data for 4D-input (batches of frames)
    else:        
        train_generator = dh.prepare_train_image_generator(df_train, train=True, val=False, test=False)
        val_generator = dh.prepare_val_image_generator(df_val, train=False, val=True, test=False)
        test_generator = dh.prepare_test_image_generator(df, train=False, val=False, test=True)
        eval_generator = dh.prepare_eval_image_generator(df, train=False, val=False, test=True)

    # Train the model
    model = train(model, args, nb_train_samples, nb_val_samples, VAL_FRACTION,
                  generator=train_generator, val_generator=val_generator)

    # Get test predictions
    y_preds, scores = ev.test(model, args, test_generator, eval_generator, nb_test_samples)

    # Get the ground truth for the test set
    y_test = df[df['Train'] == 0]['Pain'].values

    # Evaluate the model's performance
    ev.evaluate(model, y_test, y_preds, scores, args)

if __name__ == '__main__':

    # Parse the command line arguments
    arg_parser = arg_parser.ArgParser(len(sys.argv))
    args = arg_parser.parse()

    # Run the whole program, from preparing the data to evaluating
    # the model's test performance
    run()
