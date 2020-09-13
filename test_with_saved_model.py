import tensorflow as tf
import pandas as pd
import numpy as np
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
    return tf.keras.models.load_model(file_name)


def run():
    dh = DataHandler(kwargs.data_path, kwargs.of_path, 
                     kwargs.data_path + 'overview.csv',
                     ['Pain'],  # Here one can append f. ex. 'Observer'
                     (kwargs.input_width, kwargs.input_height),
                     kwargs.seq_length, kwargs.seq_stride,
                     kwargs.batch_size, COLOR,
                     kwargs.nb_labels,
                     kwargs.aug_flip, kwargs.aug_crop, kwargs.aug_light,
                     kwargs.nb_input_dims)
    
    subject_ids = pd.read_csv(kwargs.subjects_overview)['Subject'].values
    ev = Evaluator(True, True, True, True, TARGET_NAMES, kwargs.batch_size)

    # Read or create the per-subject dataframes listing all the frame paths and labels.
    subject_dfs = read_or_create_subject_dfs(dh, subject_ids)

    if '2stream' in kwargs.model or kwargs.data_type == 'of':
        subject_dfs = read_or_create_subject_rgb_and_OF_dfs(dh=dh,
                                                            subject_ids=subject_ids,
                                                            subject_dfs=subject_dfs)

    train_subjects = ast.literal_eval(kwargs.train_subjects)
    test_subjects = ast.literal_eval(kwargs.test_subjects)


    print('Horses to train on: ', train_subjects)
    print('Horses to test on: ', test_subjects)

    # Set the train-column to 1 (train), 2 (val) or 0 (test).
    if kwargs.val_fraction == 0:
        print("Using separate subject validation.")
        val_subjects = ast.literal_eval(kwargs.val_subjects)
        print('Horses to validate on: ', val_subjects)
        subject_dfs = set_train_val_test_in_df(train_subjects, val_subjects, test_subjects, subject_dfs)

    if kwargs.val_fraction == 1:
        print("Using validation fractinon.")
        print("Val fract: ", VAL_FRACTION)
        subject_dfs = set_train_test_in_df(train_subjects, test_subjects, subject_dfs)

    # Put all the separate subject-dfs into one DataFrame.
    df = pd.concat(subject_dfs)

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
            generators = get_data_2stream_5d_input(dh=dh,
                                                   df_train_rgbof=df_train,
                                                   df_val_rgbof=df_val,
                                                   df_test_rgbof=df_test)
        else:
            print('5d input model')
            if kwargs.data_type == 'rgb':
                print('Only RGB data')
            if kwargs.data_type == 'of':
                print('OF INPUT ONLY')
            generators = get_data_5d_input(dh=dh,
                                           data_type=kwargs.data_type,
                                           df_train=df_train,
                                           df_val=df_val,
                                           df_test=df_test)

    if kwargs.nb_input_dims == 4:
        if '2stream' in kwargs.model:
            print('4d input 2stream model')
            generators = get_data_2stream_4d_input(dh=dh,
                                                   df_train_rgbof=train_subjects,
                                                   df_val_rgbof=test_subjects,
                                                   df_test_rgbof=val_subjects)

        else:
            print('4d input model')
            if kwargs.data_type == 'rgb':
                print('Only RGB data')
            if kwargs.data_type == 'of':
                print('OF INPUT ONLY')

            generators = get_data_4d_input(dh,
                                           kwargs.data_type,
                                           df_train=df_train,
                                           df_val=df_val,
                                           df_test=df_test)

    train_generator, val_generator, test_generator, eval_generator = generators
    # TEMP
    start = time.time()
    test_steps, y_batches, y_batches_paths = compute_steps.compute_steps(df_test, train=False, kwargs=kwargs)
    end = time.time()
    print('Took {} s to compute testing steps'.format(end - start))

    model = load_model(model_fn)

    # Get test predictions
    y_preds, scores = ev.test(model, kwargs, test_generator, eval_generator, test_steps)

    if kwargs.nb_input_dims == 5:
        # Get the ground truth for the test set
        y_test = np.array(y_batches)  # Now in format [nb_batches, batch_size, seq_length, nb_classes]
        y_test_paths = np.array(y_batches_paths)

        if kwargs.test_run == 1:
            nb_batches = int(y_preds.shape[0]/kwargs.batch_size)
            nb_total = nb_batches * kwargs.batch_size * kwargs.seq_length
            y_test = y_test[:nb_total]
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=kwargs.nb_labels)
            y_test = np.reshape(y_test, (nb_batches*kwargs.batch_size,
                                         kwargs.seq_length,
                                         kwargs.nb_labels))
            # y_test = y_test[:nb_batches]
        else:
            nb_batches = y_test.shape[0]
            # Make 3D
            y_test = np.reshape(y_test, (nb_batches*kwargs.batch_size,
                                         kwargs.seq_length,
                                         kwargs.nb_labels))
            y_test_paths = np.reshape(y_test_paths, (nb_batches*kwargs.batch_size,
                                                     kwargs.seq_length))

    if kwargs.nb_input_dims == 4:
        y_test = np.array(y_batches)

    # Put y_preds into same format as y_test, first take the max probabilities.
    if kwargs.nb_input_dims == 5:
        y_preds_argmax = np.argmax(y_preds, axis=2)
        tesst = np.array([x.shape for x in y_preds])
        y_preds_argmax = np.array([tf.keras.utils.to_categorical(x,
                                   num_classes=kwargs.nb_labels) for x in y_preds_argmax])
    
    if kwargs.nb_input_dims == 4:
        y_preds_argmax = np.argmax(y_preds, axis=1)
        if kwargs.test_run == 1:
            y_test = y_test[:len(y_preds)]
    # Evaluate the model's performance
    ev.set_test_set(df_test)
    ev.evaluate(model=model, y_test=y_test, y_pred=y_preds_argmax,
                softmax_predictions=y_preds, scores=scores, args=kwargs, y_paths=y_test_paths)
    # # Get the ground truth for the test set
    # # y_test = df_test.values
    # y_test = np.array(y_batches)  # Now in format [nb_batches, batch_size, seq_length, nb_classes]
    # nb_batches = y_test.shape[0]
    # # Make 3D
    # y_test = np.reshape(y_test, (nb_batches*kwargs.batch_size, kwargs.seq_length, kwargs.nb_labels))

    # # Put y_preds into same format as y_test, take the max probabilities.
    # y_preds = np.argmax(y_preds, axis=2)
    # y_preds = np.array([np_utils.to_categorical(x, num_classes=kwargs.nb_labels) for x in y_preds])

    # # Evaluate the model's performance
    # ev.evaluate(model, y_test, y_preds, scores, kwargs)


if __name__ == '__main__':
    # Parse the command line arguments
    arg_parser = arg_parser.ArgParser(len(sys.argv))
    kwargs = arg_parser.parse()
    # model_fn = 'models/BEST_MODEL_2stream_5d_adadelta_LSTMunits_32_CONVfilters_16_add_v4_t5_5hl_128jpg2fps_seq10_bs8_MAG_adadelta_noaug_run2.h5'
    # model_fn = 'models/BEST_MODEL_rodriguez_adadelta_LSTMunits_1024_CONVfilters_None_jpg128_2fps_val4_t0_seq10ss10_4hl_1024ubs16_no_aug_run1.h5'
    # model_fn = 'models/BEST_MODEL_rodriguez_adadelta_LSTMunits_1024_CONVfilters_None_jpg128_2fps_val4_t1_seq10ss10_4hl_1024ubs16_no_aug_run1.h5'
    model_fn = 'models/BEST_MODEL_2stream_rmsprop_LSTMunits_None_CONVfilters_None_320x180jpg1fps_bs40_rmsprop_all_aug_v4_t1_run1.h5'
    # Run the whole program, from preparing the data to evaluating
    # the model's test performance
    run()
