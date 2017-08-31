import _pickle as cp
import pandas as pd
import sys
import ast
import os

from test_and_eval import Evaluator
from data_handler import DataHandler
from train import train
import arg_parser
import models

TARGET_NAMES = ['NO_PAIN', 'PAIN']
BATCH_SIZE = 1
COLOR = True


def run(args):
    seq_length = 50
    model = models.Model(args.model, (args.input_width, args.input_height), seq_length, args.optimizer,
                         args.lr, args.nb_lstm_units, args.nb_conv_filters, args.kernel_size,
                         args.nb_labels, args.dropout_rate, BATCH_SIZE)
    dh = DataHandler(args.data_path, (args.input_width, args.input_height), seq_length, BATCH_SIZE, COLOR)
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
    nb_train_samples = len(df[df['Train'] == 1])
    import ipdb;
    ipdb.set_trace()
    # Prepare the training and testing data
    train_generator = dh.prepare_image_generators(df, train=True)
    test_generator = dh.prepare_image_generators(df, train=False)
    import ipdb; ipdb.set_trace()

    # X_train_batch = make_batches(X_train, BATCH_SIZE)

    # Train the model
    model = train(model, args, BATCH_SIZE, nb_train_samples, generator=train_generator)

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
