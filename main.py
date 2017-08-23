import sys

from test_and_eval import Evaluator
from data_handler import DataHandler, make_batches
from train import train
import arg_parser
import models

TARGET_NAMES = ['NO_PAIN', 'PAIN']
BATCH_SIZE = 100
COLOR = True


def run(args):
    seq_length = 100
    model = models.Model(args.model, (args.input_width, args.input_height), seq_length, args.optimizer,
                         args.lr, args.nb_lstm_units, args.nb_conv_filters, args.kernel_size,
                         args.nb_labels, args.dropout_rate)
    dh = DataHandler(args.data_path, (args.input_width, args.input_height), BATCH_SIZE, COLOR)
    ev = Evaluator('cr', TARGET_NAMES)

    # dh.folders_to_csv()
    df = dh.folders_to_df()

    # Prepare the training and testing data
    X_train, y_train, X_test, y_test = dh.prepare_train_test(df)
    # X_train_batch = make_batches(X_train, BATCH_SIZE)

    # Train the model
    model = train(model, args, X_train, y_train, batch_size=BATCH_SIZE)

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
