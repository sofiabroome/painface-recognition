import pandas as pd
import wandb
import sys
import re

from test_and_eval import run_evaluation
from data_handler import DataHandler
import arg_parser
import helpers
import models
import train


# Prettyprint for dataframes with long values (filenames).
pd.set_option('max_colwidth', 800)


def run():

    dh = DataHandler(data_columns=['pain'],  # or e.g., 'observer',
                     config_dict=config_dict,
                     all_subjects_df=all_subjects_df)

    df_train, df_val, df_test = dh.get_data_indices(args)

    train_dataset, val_dataset, test_dataset = dh.get_datasets(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test)

    train_steps, _, _ = dh.get_nb_steps(df_train, 'train')

    test_steps, y_batches, y_batches_paths = dh.get_nb_steps(
        df_test, 'test')

    if config_dict['val_mode'] == 'no_val':
        val_steps = 0
    else:
        val_steps, _, _ = dh.get_nb_steps(df_val, 'val')

    if args.test_run == 1:
        config_dict['nb_epochs'] = 1
        train_steps = 2
        val_steps = 2
        test_steps = 2
        y_batches = y_batches[:test_steps]
        y_batches_paths = y_batches_paths[:test_steps]

    # Train the model

    model = models.MyModel(config_dict=config_dict)
    best_model_path = train.train(model_instance=model,
                                  config_dict=config_dict,
                                  train_steps=train_steps,
                                  val_steps=val_steps,
                                  train_dataset=train_dataset,
                                  val_dataset=val_dataset)

    if config_dict['do_evaluate']:
        run_evaluation(args=args,
                       config_dict=config_dict,
                       model=model,
                       model_path=best_model_path,
                       test_dataset=test_dataset,
                       test_steps=test_steps,
                       y_batches=y_batches,
                       y_batches_paths=y_batches_paths)

if __name__ == '__main__':

    arg_parser = arg_parser.ArgParser(len(sys.argv))
    args = arg_parser.parse()
    train_subjects = re.split('/', args.train_subjects)
    test_subjects = re.split('/', args.test_subjects)

    print('Subjects to train on: ', train_subjects)
    print('Subjects to test on: ', test_subjects)

    config_dict_module = helpers.load_module(args.config_file)
    config_dict = config_dict_module.config_dict
    if config_dict['val_mode'] == 'no_val':
        assert (config_dict['train_mode'] == 'low_level'), \
                'no_val requires low level train mode'
    config_dict['train_subjects'] = train_subjects
    config_dict['test_subjects'] = test_subjects
    wandb.init(project='pfr', config=config_dict)

    config_dict['job_identifier'] = args.job_identifier
    print('Job identifier: ', args.job_identifier)

    all_subjects_df = pd.read_csv(args.subjects_overview)

    # Run the whole program, from preparing the data to evaluating
    # the model's test performance
    run()
