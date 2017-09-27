import pandas as pd
import numpy as np
import keras
import sys
import ast

from main import set_train_test_in_df, set_train_val_test_in_df, get_data_2stream_5d_input
from main import get_data_5d_input, get_data_4d_input, get_rgb_of_dataframes
from main import get_data_2stream_4d_input
from main import df_val_split, read_or_create_horse_dfs
from data_handler import DataHandler, shuffle_blocks
from test_and_eval import Evaluator
import arg_parser

TARGET_NAMES = ['NO_PAIN', 'PAIN']
VAL_FRACTION = 0.2
COLOR = True


def load_model(file_name):
    return keras.models.load_model(file_name)


def run():
    dh = DataHandler(args.data_path, (args.input_width, args.input_height),
                         args.seq_length, args.batch_size, COLOR, args.nb_labels)
    ev = Evaluator(True, True, True, TARGET_NAMES, args.batch_size)

    # Read or create the per-horse dataframes listing all the frame paths and labels.
    horse_dfs = read_or_create_horse_dfs(dh)

    train_horses = ast.literal_eval(args.train_horses)
    test_horses = ast.literal_eval(args.test_horses)

    # Set the train-column to 1 (train), 2 (val) or 0 (test).
    if args.val_fraction == 0:
        print("Using separate horse validation.")
        val_horses = ast.literal_eval(args.val_horses)
        print('Horses to validate on: ', val_horses)
        horse_dfs = set_train_val_test_in_df(train_horses, val_horses, test_horses, horse_dfs)

    if args.val_fraction == 1:
        print("Using validation fractinon.")
        print("Val fract: ", VAL_FRACTION)
        horse_dfs = set_train_test_in_df(train_horses, test_horses, horse_dfs)

    # Put all the separate horse-dfs into one DataFrame.
    df = pd.concat(horse_dfs)

    # Shuffle the different sequences (like 1_1a_1) so that they don't always
    # appear in the same order. Also done in training generator but we do it here so that
    # the validation set is more random as well.
    df = shuffle_blocks(df)
    print("Total length of dataframe:", len(df))

    # Split training data so there is a held out validation set.

    if args.val_fraction == 1:
        df_train, df_val = df_val_split(df,
                                        val_fraction=VAL_FRACTION,
                                        batch_size=args.batch_size,
                                        round_to_batch=args.round_to_batch)
    if args.val_fraction == 0:
        df_train = df[df['Train'] == 1]
        df_val = df[df['Train'] == 2]

    print("Lengths dftr and df val:", len(df_train), len(df_val))
    df_test = df[df['Train'] == 0]
    # Count the number of samples in each partition of the data.
    nb_train_samples = len(df_train)
    nb_val_samples = len(df_val)
    nb_test_samples = len(df_test)

    # Prepare the training and testing data, format depends on model.
    # (5D/4D -- 2stream/1stream)
    if args.nb_input_dims == 5:
        if '2stream' in args.model:
            print('5d input 2stream model')
            if args.val_fraction == 0:
                generators = get_data_2stream_5d_input(dh, horse_dfs, train_horses, test_horses, val_horses)
            if args.val_fraction == 1:
                generators = get_data_2stream_5d_input(dh, horse_dfs, train_horses, test_horses)
        else:
            print('5d input model')
            if args.data_type == 'rgb':
                generators = get_data_5d_input(dh, args.data_type, df_train, df_test, df_val)
            if args.data_type == 'of':
                print('OF INPUT ONLY')
                if args.val_fraction == 0:
                    df_train, df_val, df_test = get_rgb_of_dataframes(dh, horse_dfs, train_horses, test_horses,
                                                                      val_horses)
                if args.val_fraction == 1:
                    df_train, df_val, df_test = get_rgb_of_dataframes(dh, horse_dfs, train_horses, test_horses)
                generators = get_data_5d_input(dh, args.data_type, df_train, df_test, df_val)
    if args.nb_input_dims == 4:
        if '2stream' in args.model:
            print('4d input 2stream model')
            generators = get_data_2stream_4d_input(dh, horse_dfs, train_horses, test_horses, val_horses)
        else:
            print('4d input model')
            generators = get_data_4d_input(dh, df_train, df_test, df_val)

    train_generator, val_generator, test_generator, eval_generator = generators

    model = load_model(model_fn)

    # Get test predictions
    y_preds, scores = ev.test(model, args, test_generator, eval_generator, nb_test_samples)

    # Get the ground truth for the test set
    y_test = df[df['Train'] == 0]['Pain'].values

    # Evaluate the model's performance
    ev.evaluate(model, y_test, y_preds, scores, args)


if __name__ == '__main__':
    model_fn = 'BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_64_CONVfilters_16_jpg_val4_t0_seq20.h5'
    # model_fn = 'BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_64_CONVfilters_16_jpg_val4_t1_seq10_4conv.h5'
    # model_fn = 'BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_64_CONVfilters_16_jpg_val4_t1_seq20.h5'
    # model_fn = 'BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_64_CONVfilters_16_jpg_val4_t2_seq10_4conv.h5'
    # model_fn = 'BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_64_CONVfilters_16_jpg_val4_t2_seq10_4conv_of.h5'
    # model_fn = 'BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_64_CONVfilters_16_jpg_val4_t2_seq20_3convpool.h5'
    # model_fn = 'BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_64_CONVfilters_16_jpg_val4_t2_seq20.h5'
    # model_fn = 'BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_64_CONVfilters_16_jpg_val4_t3_seq10_4conv.h5'
    # model_fn = 'BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_64_CONVfilters_16_jpg_val4_t3_seq10_4conv_of.h5'
    # model_fn = 'BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_64_CONVfilters_16_jpg_val4_t3_seq20_3convpool.h5'
    # model_fn = 'BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_64_CONVfilters_16_jpg_val4_t3_seq20.h5'

    # Parse the command line arguments
    arg_parser = arg_parser.ArgParser(len(sys.argv))
    args = arg_parser.parse()

    # Run the whole program, from preparing the data to evaluating
    # the model's test performance
    run()
