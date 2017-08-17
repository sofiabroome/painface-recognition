import sys

from test_and_eval import Evaluator
from data_handler import DataHandler
from train import train
import arg_parser
import models

target_names = ['NO_PAIN', 'PAIN']


def run():
    model = models.Model(model_args)
    dh = DataHandler(data_path)
    ev = Evaluator('cr', target_names)

    # Prepare the training and testing data
    X_train, y_train, X_test, y_test = dh.prepare_data()

    # Train the model
    model = train(model, train_args, X_train, y_train)

    # Get test predictions
    y_preds = ev.test(model, X_test)

    # Evaluate the model's performance
    ev.evaluate(model, y_preds, y_test, eval_args)

if __name__ == '__main__':

    # Parse the command line arguments
    arg_parser = arg_parser.ArgParser(len(sys.argv))
    args = arg_parser.parse()
    data_path = args[0]
    model_args = args[1:5]
    train_args = args[5:10]
    eval_args = args[10:]

    # Run the whole program, from preparing the data to evaluating
    # the model's test performance
    run()
