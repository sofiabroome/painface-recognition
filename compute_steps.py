from main import df_val_split
from keras.utils import np_utils
import pandas as pd
import numpy as np
import arg_parser
import sys


def compute_steps(df, train, kwargs):
    """
    Computes the ultimate number of valid steps given that sometimes the last
    sequence doesn't fit within the same video clip.
    :param df: pd.DataFrame
    :param train: Boolean
    :param kwargs: command line flags
    :return: int
    """

    nb_steps = 0  # AKA global number of batches.
    batch_index = 0  # Keeps track of number of samples put in batch so far.
    seq_index = 0

    nb_frames = len(df)
    print("LEN DF, in compute_steps(): ", nb_frames)

    ws = kwargs.seq_length  # "Window size" in a sliding window.
    ss = kwargs.seq_stride  # Provide argument for slinding w. stride.
    valid = nb_frames - (ws - 1)
    nw = valid // ss  # Number of windows

    y_batches = []  # List where to put all the y_arrays generated.
    y_batches_paths = []

    nb_aug = kwargs.aug_flip + kwargs.aug_crop + kwargs.aug_light
    batch_requirement = 1 + nb_aug  # Normal sequence plus augmented sequences.
    assert (kwargs.batch_size % batch_requirement) == 0

    for window_index in range(nw):
        start = window_index * ss
        stop = start + ws
        rows = df.iloc[start:stop]  # A new dataframe for the window in question.

        y_seq_list = []
        y_seq_list_paths = []

        for index, row in rows.iterrows():
            vid_seq_name = row['Video_ID']

            if index == 0:
                print('First frame. Set oldname=vidname.')
                old_vid_seq_name = vid_seq_name

            if vid_seq_name != old_vid_seq_name:
                seq_index = 0
                # print('New sequence. Settin seq ind to 0 and start on new.')
                old_vid_seq_name = vid_seq_name
                break  # In that case want to jump to the next window.
            y = row['Pain']
            y_path = row['Path']
            y_seq_list.append(y)
            y_seq_list_paths.append(y_path)
            seq_index += 1
        if batch_index == 0:
            y_batch_list = []
            y_batch_list_paths = []

        if seq_index == kwargs.seq_length:
            # Everytime a full sequence is amassed, we reset the seq_ind,
            # and increment the batch_ind.
            y_batch_list.append(y_seq_list)
            y_batch_list_paths.append(y_seq_list_paths)
            
            seq_index = 0
            batch_index += 1
            if train and (kwargs.aug_flip==1):
                y_batch_list.append(y_seq_list)
                y_batch_list_paths.append(y_seq_list_paths)
                batch_index += 1
            if train and (kwargs.aug_crop==1):
                y_batch_list.append(y_seq_list)
                y_batch_list_paths.append(y_seq_list_paths)
                batch_index += 1
            if train and (kwargs.aug_light==1):
                y_batch_list.append(y_seq_list)
                y_batch_list_paths.append(y_seq_list_paths)
                batch_index += 1

        if batch_index % kwargs.batch_size == 0 and not batch_index == 0:
            y_array = np.array(y_batch_list, dtype=np.uint8)
            if kwargs.nb_labels == 2:
                y_array = np_utils.to_categorical(y_array, num_classes=kwargs.nb_labels)
                y_array = np.reshape(y_array, (kwargs.batch_size, kwargs.seq_length, kwargs.nb_labels))
            nb_steps += 1
            batch_index = 0
            y_batches.append(y_array)
            y_batches_paths.append(y_batch_list_paths)
    return nb_steps, y_batches, y_batches_paths


if __name__ == '__main__':
    arg_parser = arg_parser.ArgParser(len(sys.argv))
    args = arg_parser.parse()
    VAL_FRACTION = 0.3

    df_train = pd.read_csv('data/train_frames.csv')
    df_test = pd.read_csv('data/test_frames.csv')
    df_train, df_val = df_val_split(df_train,
                                    val_fraction=VAL_FRACTION,
                                    batch_size=args.batch_size,
                                    round_to_batch=args.round_to_batch)

    train_steps = compute_steps(df_train, args)
    test_steps = compute_steps(df_test, args)

