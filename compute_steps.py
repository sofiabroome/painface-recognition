import tensorflow as tf
import numpy as np


def compute_steps(df, train, config_dict):
    """
    Computes the ultimate number of valid steps given that sometimes the last
    sequence doesn't fit within the same video clip.
    :param df: pd.DataFrame
    :param train: Boolean
    :param config_dict: dict
    :return: int
    """

    nb_steps = 0  # AKA global number of batches.
    batch_index = 0  # Keeps track of number of samples put in batch so far.
    seq_index = 0

    nb_frames = len(df)

    window_size = config_dict['seq_length']
    window_stride = config_dict['seq_stride']
    last_valid_start_index = nb_frames - (window_size - 1)
    last_valid_end_index = last_valid_start_index + (window_size-1)
    number_of_windows = last_valid_end_index // window_stride

    assert (number_of_windows >= config_dict['batch_size'])

    y_batches = []  # List where to put all the y_arrays generated.
    y_batches_paths = []

    nb_aug = config_dict['aug_flip'] + config_dict['aug_crop'] + config_dict['aug_light']
    batch_requirement = 1 + nb_aug  # Normal sequence plus augmented sequences.
    assert (config_dict['batch_size'] % batch_requirement) == 0

    for window_index in range(number_of_windows):
        start = window_index * window_stride
        stop = start + window_size
        rows = df.iloc[start:stop]  # A new dataframe for the window in question.

        y_seq_list = []
        y_seq_list_paths = []

        for index, row in rows.iterrows():
            vid_seq_name = row['video_id']

            if index == 0:
                old_vid_seq_name = vid_seq_name

            if vid_seq_name != old_vid_seq_name:
                seq_index = 0
                old_vid_seq_name = vid_seq_name
                break  # In that case want to jump to the next window.
            y = row['pain']
            y_path = row['path']
            y_seq_list.append(y)
            y_seq_list_paths.append(y_path)
            seq_index += 1
        if batch_index == 0:
            y_batch_list = []
            y_batch_list_paths = []

        if seq_index == config_dict['seq_length']:
            # Everytime a full sequence is amassed, we reset the seq_ind,
            # and increment the batch_ind.
            y_batch_list.append(y_seq_list)
            y_batch_list_paths.append(y_seq_list_paths)
            
            seq_index = 0
            batch_index += 1
            if train and (config_dict['aug_flip'] == 1):
                y_batch_list.append(y_seq_list)
                y_batch_list_paths.append(y_seq_list_paths)
                batch_index += 1
            if train and (config_dict['aug_crop'] == 1):
                y_batch_list.append(y_seq_list)
                y_batch_list_paths.append(y_seq_list_paths)
                batch_index += 1
            if train and (config_dict['aug_light'] == 1):
                y_batch_list.append(y_seq_list)
                y_batch_list_paths.append(y_seq_list_paths)
                batch_index += 1

        if batch_index % config_dict['batch_size'] == 0 and not batch_index == 0:
            y_array = np.array(y_batch_list, dtype=np.uint8)
            if config_dict['nb_labels'] == 2:
                y_array = tf.keras.utils.to_categorical(y_array,
                                                        num_classes=config_dict['nb_labels'])
                y_array = np.reshape(y_array,
                                     (config_dict['batch_size'],
                                      config_dict['seq_length'],
                                      config_dict['nb_labels']))
            nb_steps += 1
            batch_index = 0
            y_batches.append(y_array)
            y_batches_paths.append(y_batch_list_paths)

    return nb_steps, y_batches, y_batches_paths

