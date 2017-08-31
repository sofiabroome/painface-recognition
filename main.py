import _pickle as cp
import pandas as pd
import numpy as np
import sys
import ast
import os

from test_and_eval import Evaluator
from data_handler import DataHandler, shuffle_blocks
from train import train
import arg_parser
import models

TARGET_NAMES = ['NO_PAIN', 'PAIN']
BATCH_SIZE = 500
COLOR = True

np.random.seed(100)


def df_val_split(df, val_fraction, batch_size, round_to_batch=True):
    df = df.loc[df['Train'] == 1]
    if round_to_batch:
        ns = len(df)
        num_val = int(val_fraction * ns - val_fraction * ns % batch_size)
        df_val = df.iloc[-num_val:, :]
        df_train = df.iloc[:-num_val, :]

    return df_train, df_val


def run(args):
    seq_length = 50
    model = models.Model(args.model, (args.input_width, args.input_height),
                         seq_length, args.optimizer, args.lr, args.nb_lstm_units,
                         args.nb_conv_filters, args.kernel_size,
                         args.nb_labels, args.dropout_rate, BATCH_SIZE)
    dh = DataHandler(args.data_path, (args.input_width, args.input_height),
                     seq_length, BATCH_SIZE, COLOR, args.nb_labels)
    ev = Evaluator('cr', TARGET_NAMES)

    # dh.folders_to_csv()
    horse_dfs = []
    for horse in range(1,7):
        horse_csv_path = 'horse_' + str(horse) + '.csv'
        if os.path.isfile(horse_csv_path):
            hdf = pd.read_csv(horse_csv_path)
        else:
            print('Making a DataFrame for horse id: ', horse)
            hdf = dh.horse_to_df(horse)
        horse_dfs.append(hdf)

    train_horses = ast.literal_eval(args.train_horses)
    test_horses = ast.literal_eval(args.test_horses)

    # Set the train-column to 1 (yes) or 0 (no).
    for trh in train_horses:
        horse_dfs[trh]['Train'] = 1

    for teh in test_horses:
        horse_dfs[teh]['Train'] = 0
    # Put all the separate horse-dfs into one DataFrame.
    df = pd.concat(horse_dfs)
    df = shuffle_blocks(df)

    df_train, df_val = df_val_split(df, val_fraction=0.1, batch_size=BATCH_SIZE, round_to_batch=True)
    nb_train_samples = len(df_train)
    nb_val_samples = len(df_val)
    # Prepare the training and testing data
    train_generator = dh.prepare_image_generators(df_train, train=True, val=False, test=False)
    val_generator = dh.prepare_image_generators(df_val, train=False, val=True, test=False)
    test_generator = dh.prepare_image_generators(df, train=False, val=False, test=True)

    # X_train_batch = make_batches(X_train, BATCH_SIZE)

    # Train the model
    model = train(model, args, BATCH_SIZE, nb_train_samples, nb_val_samples,
                  generator=train_generator, val_generator=val_generator)

    # # Get test predictions
    # y_preds = ev.test(model, X_test)
    #
    # # Evaluate the model's performance
    # ev.evaluate(model, y_preds, y_test, eval_args)

if __name__ == '__main__':

    # Parse the command line arguments
    arg_parser = arg_parser.ArgParser(len(sys.argv))
    args = arg_parser.parse()

    # Run the whole program, from preparing the data to evaluating
    # the model's test performance
    run(args)
