import sys

from test_and_eval import Evaluator
from data_handler import DataHandler
from train import train
import arg_parser
import models

target_names = ['NO_PAIN', 'PAIN']


def run(args):
    seq_length = 100
    model = models.Model(args.model, (args.input_width, args.input_height), seq_length, args.optimizer,
                         args.lr, args.nb_lstm_units, args.nb_conv_filters, args.kernel_size,
                         args.nb_labels, args.dropout_rate)
    dh = DataHandler(args.data_path, (args.input_width, args.input_height))
    ev = Evaluator('cr', target_names)

    dh.folders_to_csv()

    # Prepare the training and testing data
    # X_train, y_train, X_test, y_test = dh.prepare_data()
    #
    # # Train the model
    # model = train(model, train_args, X_train, y_train)
    #
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
