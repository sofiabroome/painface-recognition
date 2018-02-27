from keras.utils import np_utils
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import time
import sys
import ast
import os

from data_handler import DataHandler, shuffle_blocks
from test_and_eval import Evaluator
import compute_steps
import arg_parser

TARGET_NAMES = ['NO_PAIN', 'PAIN']
VAL_FRACTION = 0.2
COLOR = True

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def load_model(file_name):
    return keras.models.load_model(file_name)


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
        print(kwargs.data_path)
        horse_csv_path = kwargs.data_path + 'horse_' + str(horse) + '.csv'
        if os.path.isfile(horse_csv_path):
            hdf = pd.read_csv(horse_csv_path)
        else:
            print('Making a DataFrame for horse id: ', horse)
            hdf = dh.horse_to_df(horse)
            hdf.to_csv(path_or_buf=horse_csv_path)
        horse_dfs.append(hdf)
    return horse_dfs


def prepare_rgb_of_dataframe(dh, horse_dfs, train_horses, test_horses, val_horses=None):
    horse_rgb_OF_dfs = read_or_create_horse_rgb_and_OF_dfs(dh, horse_dfs)
    if kwargs.val_fraction == 0:
        horse_rgb_OF_dfs = set_train_val_test_in_df(train_horses, val_horses, test_horses, horse_rgb_OF_dfs)
    else:
        horse_rgb_OF_dfs = set_train_test_in_df(train_horses, test_horses, horse_rgb_OF_dfs)
    df_rgb_and_of = pd.concat(horse_rgb_OF_dfs)
    df_rgb_and_of = shuffle_blocks(df_rgb_and_of, 'Video_ID')
    return df_rgb_and_of


def read_or_create_horse_rgb_and_OF_dfs(dh, horse_dfs):
    # Read or create the per-horse optical flow files listing all the frame paths and labels.
    horse_rgb_OF_dfs = []
    for horse_id in range(1, 7):
        print(kwargs.data_path)
        horse_of_csv_path = dh.of_path + '/horse_' + str(horse_id) + '.csv'
        if os.path.isfile(horse_of_csv_path):
            hdf = pd.read_csv(horse_of_csv_path)
        else:
            print('Making a DataFrame for horse id: ', horse_id)
            hdf = dh.save_OF_paths_to_df(horse_id, horse_dfs[horse_id-1])
            hdf.to_csv(path_or_buf=horse_of_csv_path)
        horse_rgb_OF_dfs.append(hdf)
    return horse_rgb_OF_dfs


def set_train_val_test_in_df(train_horses, val_horses, test_horses, dfs):
    for trh in train_horses:
        dfs[trh]['Train'] = 1

    for vh in val_horses:
        dfs[vh]['Train'] = 2

    for teh in test_horses:
        dfs[teh]['Train'] = 0
    return dfs


def set_train_test_in_df(train_horses, test_horses, dfs):
    for trh in train_horses:
        dfs[trh]['Train'] = 1

    for teh in test_horses:
        dfs[teh]['Train'] = 0
    return dfs


def get_data_4d_input(dh, data_type, df_train, df_test, df_val=None):
    """
    Prepare the training and testing data for 4D-input (batches of frames)
    """
    train_generator = dh.prepare_image_generator(df_train, data_type, train=True, val=False, test=False, evaluate=False)
    val_generator = dh.prepare_image_generator(df_val, data_type, train=False, val=True, test=False, evaluate=False)
    test_generator = dh.prepare_image_generator(df_test, data_type, train=False, val=False, test=True, evaluate=False)
    eval_generator = dh.prepare_image_generator(df_test, data_type, train=False, val=False, test=False,  evaluate=True)
    generators = (train_generator, val_generator, test_generator, eval_generator)
    return generators


def get_data_5d_input(dh, data_type, df_train, df_val, df_test):
    """
    Prepare the training and testing data for 5D-input (batches of sequences of frames)
    """
    train_generator = dh.prepare_image_generator_5D(df_train,
                                                    data_type=data_type,
                                                    train=True,
                                                    val=False, test=False, eval=False)
    val_generator = dh.prepare_image_generator_5D(df_val,
                                                  data_type=data_type,
                                                  train=False,
                                                  val=True, test=False, eval=False)
    test_generator = dh.prepare_image_generator_5D(df_test,
                                                   data_type=data_type,
                                                   train=False,
                                                   val=False, test=True, eval=False)
    eval_generator = dh.prepare_image_generator_5D(df_test,
                                                   data_type=data_type,
                                                   train=False,
                                                   val=False, test=False, eval=True)
    generators = (train_generator, val_generator, test_generator, eval_generator)
    return generators


def get_data_2stream_4d_input(dh, df_train_rgbof, df_val_rgbof, df_test_rgbof):
    train_generator = dh.prepare_generator_2stream(df_train_rgbof, train=True,
                                                   val=False, test=False, evaluate=False)
    val_generator = dh.prepare_generator_2stream(df_val_rgbof, train=False,
                                                 val=True, test=False, evaluate=False)
    test_generator = dh.prepare_generator_2stream(df_test_rgbof, train=False,
                                                  val=False, test=True, evaluate=False)
    eval_generator = dh.prepare_generator_2stream(df_test_rgbof, train=False,
                                                  val=False, test=False, evaluate=True)
    generators = (train_generator, val_generator, test_generator, eval_generator)
    return generators


def get_rgb_of_dataframes(dh, horse_dfs, train_horses, test_horses, val_horses=None):
    if kwargs.val_fraction == 0:
        df_rgb_and_of = prepare_rgb_of_dataframe(dh, horse_dfs, train_horses, test_horses, val_horses)
        df_train_rgbof = df_rgb_and_of[df_rgb_and_of['Train'] == 1]
        df_val_rgbof = df_rgb_and_of[df_rgb_and_of['Train'] == 2]
    if kwargs.val_fraction == 1:
        df_rgb_and_of = prepare_rgb_of_dataframe(dh, horse_dfs, train_horses, test_horses)
        df_train_rgbof, df_val_rgbof = df_val_split(df_rgb_and_of,
                                                    VAL_FRACTION,
                                                    batch_size=kwargs.batch_size,
                                                    round_to_batch=True)
    df_test_rgbof = df_rgb_and_of[df_rgb_and_of['Train'] == 0]
    return df_train_rgbof, df_val_rgbof, df_test_rgbof


def get_data_2stream_5d_input(dh, horse_dfs, train_horses, test_horses, val_horses=None):
    """
    Prepare the training and testing data for 5D-input (batches of sequences of frames)
    """
    print("2stream model of some sort.", kwargs.model)
    # Read or create the per-horse optical flow files listing all the frame paths and labels.
    df_train_rgbof, df_val_rgbof, df_test_rgbof = get_rgb_of_dataframes(dh, horse_dfs,
                                                                        train_horses,
                                                                        test_horses,
                                                                        val_horses)
    print("Using the 5D generator for 2stream")
    train_generator = dh.prepare_2stream_image_generator_5D(df_train_rgbof, train=True,
                                                            val=False, test=False, evaluate=False)
    val_generator = dh.prepare_2stream_image_generator_5D(df_val_rgbof, train=False,
                                                          val=True, test=False, evaluate=False)
    test_generator = dh.prepare_2stream_image_generator_5D(df_test_rgbof, train=False,
                                                           val=False, test=True, evaluate=False)
    eval_generator = dh.prepare_2stream_image_generator_5D(df_test_rgbof, train=False,
                                                           val=False, test=False, evaluate=True)
    generators = (train_generator, val_generator, test_generator, eval_generator)
    return generators


def run():
    dh = DataHandler(kwargs.data_path, kwargs.of_path, 
                     (kwargs.input_width, kwargs.input_height),
                     kwargs.seq_length, kwargs.seq_stride,
                     kwargs.batch_size, COLOR,
                     kwargs.nb_labels, kwargs.aug_flip, kwargs.aug_crop, kwargs.aug_light)

    ev = Evaluator(True, True, True, TARGET_NAMES, kwargs.batch_size)

    # Read or create the per-horse dataframes listing all the frame paths and labels.
    horse_dfs = read_or_create_horse_dfs(dh)

    train_horses = ast.literal_eval(kwargs.train_horses)
    test_horses = ast.literal_eval(kwargs.test_horses)

    print('Horses to train on: ', train_horses)
    print('Horses to test on: ', test_horses)

    # Set the train-column to 1 (train), 2 (val) or 0 (test).
    if kwargs.val_fraction == 0:
        print("Using separate horse validation.")
        val_horses = ast.literal_eval(kwargs.val_horses)
        print('Horses to validate on: ', val_horses)
        horse_dfs = set_train_val_test_in_df(train_horses, val_horses, test_horses, horse_dfs)

    if kwargs.val_fraction == 1:
        print("Using validation fractinon.")
        print("Val fract: ", VAL_FRACTION)
        horse_dfs = set_train_test_in_df(train_horses, test_horses, horse_dfs)

    # Put all the separate horse-dfs into one DataFrame.
    df = pd.concat(horse_dfs)

    # Shuffle the different sequences (like 1_1a_1) so that they don't always
    # appear in the same order. Also done in training generator but we do it here so that
    # the validation set is more random as well.
    df = shuffle_blocks(df, 'Video_ID')
    print("Total length of dataframe:", len(df))

    # Split training data so there is a held out validation set.

    if kwargs.val_fraction == 1:
        df_train, df_val = df_val_split(df,
                                        val_fraction=VAL_FRACTION,
                                        batch_size=kwargs.batch_size,
                                        round_to_batch=kwargs.round_to_batch)
    if kwargs.val_fraction == 0:
        df_train = df[df['Train'] == 1]
        df_val = df[df['Train'] == 2]

    df_test = df[df['Train'] == 0]
    
    # Reset all indices so they're 0->N.
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    # Count the number of samples in each partition of the data.
    nb_train_samples = len(df_train)
    nb_val_samples = len(df_val)
    nb_test_samples = len(df_test)

    print("Lengths dftr, dfval, dftest: ", nb_train_samples, nb_val_samples, nb_test_samples)

    # Prepare the training and testing data, format depends on model.
    # (5D/4D -- 2stream/1stream)
    

    if kwargs.nb_input_dims == 5:

        if '2stream' in kwargs.model:
            print('5d input 2stream model')
            if kwargs.val_fraction == 0:
                generators = get_data_2stream_5d_input(dh, horse_dfs, train_horses, test_horses, val_horses)
            if kwargs.val_fraction == 1:
                generators = get_data_2stream_5d_input(dh, horse_dfs, train_horses, test_horses)

        else:
            print('5d input model')
            if kwargs.data_type == 'rgb':
                generators = get_data_5d_input(dh, kwargs.data_type, df_train=df_train, df_val=df_val, df_test=df_test)
            if kwargs.data_type == 'of':
                print('OF INPUT ONLY')
                if kwargs.val_fraction == 0:
                    df_train, df_val, df_test = get_rgb_of_dataframes(dh, horse_dfs, train_horses, test_horses,
                                                                      val_horses)
                if kwargs.val_fraction == 1:
                    df_train, df_val, df_test = get_rgb_of_dataframes(dh, horse_dfs, train_horses, test_horses)
                generators = get_data_5d_input(dh, kwargs.data_type, df_train=df_train, df_val=df_val, df_test=df_test)

    if kwargs.nb_input_dims == 4:
        if '2stream' in kwargs.model:
            print('4d input 2stream model')
            generators = get_data_2stream_4d_input(dh, horse_dfs, train_horses, test_horses, val_horses)

        else:
            print('4d input model')
            if kwargs.data_type == 'rgb':
                generators = get_data_4d_input(dh, kwargs.data_type, df_train=df_train, df_val=df_val, df_test=df_test)
            if kwargs.data_type == 'of':
                df_train, df_val, df_test = get_rgb_of_dataframes(dh, horse_dfs, train_horses, test_horses, val_horses)
                generators = get_data_4d_input(dh, kwargs.data_type, df_train=df_train, df_test=df_test, df_val=df_val)

    train_generator, val_generator, test_generator, eval_generator = generators

    start = time.time()
    test_steps, y_batches = compute_steps.compute_steps(df_test, kwargs)
    end = time.time()
    print('Took {} s to compute testing steps'.format(end - start))

    model = load_model(model_fn)

    # Get test predictions
    y_preds, scores = ev.test(model, kwargs, test_generator, eval_generator, test_steps)

    # Get the ground truth for the test set
    # y_test = df_test.values
    y_test = np.array(y_batches)  # Now in format [nb_batches, batch_size, seq_length, nb_classes]
    nb_batches = y_test.shape[0]
    # Make 3D
    y_test = np.reshape(y_test, (nb_batches*kwargs.batch_size, kwargs.seq_length, kwargs.nb_labels))

    # Put y_preds into same format as y_test, take the max probabilities.
    y_preds = np.argmax(y_preds, axis=2)
    y_preds = np.array([np_utils.to_categorical(x, num_classes=kwargs.nb_labels) for x in y_preds])

    # Evaluate the model's performance
    ev.evaluate(model, y_test, y_preds, scores, kwargs)


if __name__ == '__main__':
    arg_parser = arg_parser.ArgParser(len(sys.argv))
    kwargs = arg_parser.parse()

    # model_fn = 'models/BEST_MODEL_2stream_5d_adam_LSTMunits_32_CONVfilters_16_concat_v4_t0_1hl_seq10_bs2_MAG.h5'

    # model_fn = 'models/BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_64_CONVfilters_16_jpg128_1fps_val4_t0_seq10_4hl_64ubs10_randomflip_aug.h5'
    # model_fn = 'models/BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_64_CONVfilters_16_jpg128_1fps_val4_t2_seq10_4hl_64ubs10_randomflip_aug.h5'
    # model_fn = 'models/BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_64_CONVfilters_16_jpg128_1fps_val4_t2_seq10_4hl_64ubs10_randomflip_aug_run2.h5' 
    # model_fn = 'models/BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_64_CONVfilters_16_jpg128_1fps_val4_t3_seq10_4hl_64ubs10_randomflip_aug_run2.h5'
    model_fn = 'models/BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_32_CONVfilters_16_jpg128_2fps_val4_t0_seq10ss5_4hl_32ubs18_randomflip_aug_run2.h5'

# Parse the command line arguments


    # Run the whole program, from preparing the data to evaluating
    # the model's test performance
    run()
