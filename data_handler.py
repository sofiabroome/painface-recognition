import tensorflow as tf
import compute_steps
import pandas as pd
import numpy as np
import random
import time
import cv2
import re
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import helpers

class DataHandler:
    def __init__(self, data_columns, config_dict, all_subjects_df):
        """
        Constructor for the DataHandler.
        :param config_dict: dict
        """
        self.data_columns = data_columns
        self.image_size = config_dict['input_width'], config_dict['input_height']
        self.batch_size = config_dict['batch_size']
        self.seq_length = config_dict['seq_length']
        self.seq_stride = config_dict['seq_stride']
        self.batch_size = config_dict['batch_size']
        self.color_channels = 3 if config_dict['color'] else 1
        self.nb_labels = config_dict['nb_labels']
        self.aug_flip = config_dict['aug_flip']
        self.aug_crop = config_dict['aug_crop']
        self.aug_light = config_dict['aug_light']
        self.nb_input_dims = config_dict['nb_input_dims']
        self.dataset_rgb_path_dict = {'pf': config_dict['pf_rgb_path'],
                                      'lps': config_dict['lps_rgb_path']}
        self.dataset_of_path_dict = {'pf': config_dict['pf_of_path'],
                                     'lps': config_dict['lps_of_path']}
        self.config_dict = config_dict
        self.all_subjects_df = all_subjects_df
        self.pixel_mean = config_dict['pixel_mean']
        self.pixel_std = config_dict['pixel_std']

    def get_dataset(self, df, train):
        """
        Get a dataset corresponding to a DataFrame, appropriate for the model.
        :param df: pd.DataFrame
        :param train:  boolean
        :return: tf.data.Dataset
        """

        if self.config_dict['nb_input_dims'] == 5:
            if '2stream' in self.config_dict['model']:
                dataset = tf.data.Dataset.from_generator(
                    lambda: self.prepare_2stream_image_generator_5D(df, train),
                    output_types=(tf.float32, tf.uint8))
            else:
                dataset = tf.data.Dataset.from_generator(
                    lambda: self.prepare_image_generator_5D(df, train),
                    output_types=(tf.float32, tf.uint8),
                    output_shapes=(
                        tf.TensorShape([None, None, self.image_size[0],
                            self.image_size[1], self.color_channels]),
                        tf.TensorShape([None, 2]))
                    )
        if self.config_dict['nb_input_dims'] == 4:
            if '2stream' in self.config_dict['model']:
                generator = self.prepare_generator_2stream(
                    df, train
                )
            else:
                generator = self.prepare_image_generator(
                    df, train
                )

        return dataset

    def df_val_split(self,
                     df,
                     val_fraction,
                     batch_size):
        """
        If args.val_mode == 'fraction', split the dataframe with training data into two parts,
        a training set and a held out validation set (the last specified fraction from the df).
        :param df: pd.Dataframe
        :param val_fraction: float
        :param batch_size: int
        :return: pd.Dataframe, pd.Dataframe
        """
        df = df.loc[df['train'] == 1]
        if self.config_dict['round_to_batch']:
            ns = len(df)
            ns_rounded = ns - ns % batch_size
            num_val = int(val_fraction * ns_rounded - val_fraction * ns_rounded % batch_size)
            df = df.iloc[:ns_rounded]
            df_val = df.iloc[-num_val:, :]
            df_train = df.iloc[:-num_val, :]

        return df_train, df_val

    def read_or_create_subject_dfs(self, subject_ids):
        """
        Read or create the per-subject dataframes listing
        all the frame paths and corresponding labels and metadata.
        :param subject_ids: list of ints referring to subjects
        :return: {str: pd.Dataframe}
        """
        subject_dfs = {}
        for ind, subject_id in enumerate(subject_ids):
            dataset = self.all_subjects_df.loc[ind]['dataset']
            path_key = 'pf_rgb_path' if dataset == 'pf' else 'lps_rgb_path'
            subject_csv_path = os.path.join(
                self.config_dict[path_key], subject_id) + '.csv'
            if os.path.isfile(subject_csv_path):
                sdf = pd.read_csv(subject_csv_path)
            else:
                print('Making a DataFrame for: ', subject_id)
                sdf = self.subject_to_df(subject_id, dataset, self.config_dict)
                sdf.to_csv(path_or_buf=subject_csv_path)
            subject_dfs[subject_id] = sdf
        return subject_dfs

    def read_or_create_subject_rgb_and_OF_dfs(self,
                                              subject_ids,
                                              subject_dfs):
        """
        Read or create the per-subject optical flow files listing
        all the frame paths and labels.
        :param subject_ids: list of ints referring to subjects
        :param subject_dfs: [pd.DataFrame]
        :return: {str: pd.Dataframe}
        """
        subject_rgb_OF_dfs = {}
        for ind, subject_id in enumerate(subject_ids):
            dataset = self.all_subjects_df.loc[ind]['dataset']
            path_key = 'pf_of_path' if dataset == 'pf' else 'lps_of_path'
            subject_of_csv_path = os.path.join(
                self.config_dict[path_key], subject_id) + '.csv'
            if os.path.isfile(subject_of_csv_path):
                sdf = pd.read_csv(subject_of_csv_path)
            else:
                print('Making a DataFrame with optical flow for: ', subject_id)
                sdf = self.save_OF_paths_to_df(subject_id,
                                               subject_dfs[subject_id],
                                               dataset=dataset)
                sdf.to_csv(path_or_buf=subject_of_csv_path)
            subject_rgb_OF_dfs[subject_id] = sdf
        return subject_rgb_OF_dfs

    def set_train_val_test_in_df(self,
                                 dfs):
        """
        Mark in input dataframe which subjects to use for train, val or test.
        Used when val_mode == 'subject'
        :param val_subjects: [int]
        :param dfs: [pd.DataFrame]
        :return: [pd.DataFrame]
        """
        for trh in self.config_dict['train_subjects']:
            dfs[trh]['train'] = 1

        if self.config_dict['val_mode'] == 'subject':
            for vh in self.config_dict['val_subjects']:
                dfs[vh]['train'] = 2

        for teh in self.config_dict['test_subjects']:
            dfs[teh]['train'] = 0
        return dfs

    def get_data_indices(self, args):
        subject_ids = self.all_subjects_df['subject'].values

        # Read the dataframes listing all the frame paths and labels
        subject_dfs = self.read_or_create_subject_dfs(subject_ids=subject_ids)

        # If we need optical flow
        if '2stream' in self.config_dict['model'] or self.config_dict['data_type'] == 'of':
            subject_dfs = self.read_or_create_subject_rgb_and_OF_dfs(
                subject_ids=subject_ids,
                subject_dfs=subject_dfs)

        # Set the train-column to 1 (train), 2 (val) or 0 (test).
        if self.config_dict['val_mode'] == 'subject':
            print("Using separate subject validation.")
            self.config_dict['val_subjects'] = re.split('/', args.val_subjects)
            print('Horses to validate on: ', self.config_dict['val_subjects'])
            subject_dfs = self.set_train_val_test_in_df(dfs=subject_dfs)

        if self.config_dict['val_mode'] == 'fraction' or \
           self.config_dict['val_mode'] == 'no_val':
            subject_dfs = self.set_train_val_test_in_df(dfs=subject_dfs)

        # Put all the separate subject-dfs into one DataFrame.
        df = pd.concat(list(subject_dfs.values()), sort=False)

        print("Total length of dataframe:", len(df))

        # Split training data so there is a held out validation set.
        if self.config_dict['val_mode'] == 'fraction':
            print("Val fract: ", self.config_dict['val_fraction_value'])
            df_train, df_val = self.df_val_split(
                df=df,
                val_fraction=self.config_dict['val_fraction_value'],
                batch_size=self.config_dict['batch_size'])
        else:
            df_train = df.loc[df['train'] == 1]

        if self.config_dict['val_mode'] == 'subject':
            df_val = df[df['train'] == 2]

        df_test = df[df['train'] == 0]

        # Reset all indices so they're 0->N.
        print('\nResetting dataframe indices...')
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        if not self.config_dict['val_mode'] == 'no_val':
            df_val.reset_index(drop=True, inplace=True)
        else:
            df_val = []

        print("Nb. of train, val and test samples: ",
              len(df_train), len(df_val), len(df_test), '\n')

        return df_train, df_val, df_test

    def get_datasets(self, df_train, df_val, df_test):
        print('\nPreparing data generators...')
        train_dataset = self.get_dataset(df_train, train=True)
        val_dataset = self.get_dataset(df_val, train=False)
        test_dataset = self.get_dataset(df_test, train=False)

        return train_dataset, val_dataset, test_dataset

    def get_nb_steps(self, df, train_str='train'):
        start = time.time()
        train_mode = True if train_str == 'train' else False
        nb_steps, y_batches, y_batches_paths = compute_steps.compute_steps(
            df, train=train_mode, config_dict=self.config_dict)
        end = time.time()
        print('\nTook {:.2f} s to compute {} {} steps'.format(
            end - start, nb_steps, train_str))

        return nb_steps, y_batches, y_batches_paths

    def prepare_generator_2stream(self, df, train):
        """
        Prepare the frames into labeled train and test sets, with help from the
        DataFrame with .jpg-paths and labels for train and pain.
        :param df: pd.DataFrame
        :param train: Boolean
        :return: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """

        nb_frames = len(df)
        print("LEN DF (nb. of frames): ", nb_frames)

        # Make sure that no augmented batches are thrown away,
        # because we really want to augment the dataset.

        nb_aug = self.aug_flip + self.aug_crop + self.aug_light
        batch_requirement = 1 + nb_aug  # Normal sequence plus augmented sequences.
        assert (self.batch_size % batch_requirement) == 0

        while True:
            # Shuffle blocks between epochs.
            if train:
                df = shuffle_blocks(df, 'video_id')
            batch_index = 0
            for index, row in df.iterrows():
                if batch_index == 0:
                    X_batch_list = []
                    y_batch_list = []
                    flow_batch_list = []

                x = self.get_image(row['path'])
                X_batch_list.append(x)
                y = row['pain']
                y_batch_list.append(y)
                flow = self.get_flow(row['of_path'])
                flow_batch_list.append(flow)

                batch_index += 1

                if train and (self.aug_flip == 1):
                    # Flip both RGB and flow
                    X_flipped = self.flip_image(x)
                    flow_flipped = self.flip_image(flow)
                    # Append to the respective batch lists
                    X_batch_list.append(X_flipped)
                    y_batch_list.append(y)
                    flow_batch_list.append(flow_flipped)
                    batch_index += 1

                if train and (self.aug_crop == 1):
                    crop_size = 99
                    # Flip both RGB and flow
                    X_cropped = self.random_crop_resize_single_image(x, crop_size, crop_size)
                    flow_cropped = self.random_crop_resize_single_image(flow, crop_size, crop_size)
                    # Append to the respective batch lists
                    X_batch_list.append(X_cropped)
                    y_batch_list.append(y)
                    flow_batch_list.append(flow_cropped)
                    batch_index += 1

                if train and (self.aug_light == 1):
                    # Flip both RGB and flow
                    X_shaded = self.add_gaussian_noise_to_single_image(x)
                    flow_shaded = self.add_gaussian_noise_to_single_image(flow)
                    # Append to the respective batch lists
                    X_batch_list.append(X_shaded)
                    y_batch_list.append(y)
                    flow_batch_list.append(flow_shaded)
                    batch_index += 1

                if batch_index % self.batch_size == 0 and not batch_index == 0:
                    X_array = np.array(X_batch_list, dtype=np.float32)
                    y_array = np.array(y_batch_list, dtype=np.uint8)
                    flow_array = np.array(flow_batch_list, dtype=np.float32)
                    if self.nb_labels == 2:
                        y_array = tf.keras.utils.to_categorical(y_array, num_classes=self.nb_labels)
                    y_array = np.reshape(y_array, (self.batch_size, self.nb_labels))
                    batch_index = 0
                    # print(X_array.shape, flow_array.shape, y_array.shape)
                    yield [X_array, flow_array], y_array

    def prepare_2stream_image_generator_5D(self, df, train):
        """
        Prepare the frames into labeled train and test sets, with help from the
        DataFrame with .jpg-paths and labels for train and pain.
        :param df: pd.DataFrame
        :param train: Boolean
        :param config_dict: dict
        :return: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """

        nb_frames = len(df)
        print("LEN DF (nb. of frames): ", nb_frames)

        ws = self.seq_length  # "Window size" in a sliding window.
        ss = self.seq_stride  # Provide argument for slinding w. stride.
        valid = nb_frames - (ws - 1)
        nw = valid//ss  # Number of windows
        print('Number of windows', nw)

        this_index = 0
        seq_index = 0

        # Make sure that no augmented sequences are thrown away,
        # because we really want to augment the dataset.

        nb_aug = self.aug_flip + self.aug_crop + self.aug_light
        batch_requirement = 1 + nb_aug  # Normal sequence plus augmented sequences.
        assert (self.batch_size % batch_requirement) == 0

        while True:
            # Shuffle blocks between epochs.
            if train:
                df = shuffle_blocks(df, 'video_id')
            batch_index = 0
            for window_index in range(nw):
                start = window_index * ss
                stop = start + ws
                rows = df.iloc[start:stop]  # A new dataframe for the window in question.

                X_seq_list = []
                y_seq_list = []
                flow_seq_list = []

                for index, row in rows.iterrows():
                    vid_seq_name = row['video_id']

                    if this_index == 0:
                        old_vid_seq_name = vid_seq_name  # This variable is set once
                        this_index += 1
                    
                    if vid_seq_name != old_vid_seq_name:
                        seq_index = 0
                        old_vid_seq_name = vid_seq_name
                        break  # Should not have seqs with mixed video IDs

                    if (seq_index % self.config_dict['rgb_period']) == 0:
                        x = self.get_image(row['path'])
                        X_seq_list.append(x)
                        y = row['pain']
                        y_seq_list.append(y)

                    if (seq_index % self.config_dict['flow_period']) == 0:
                        flow = self.get_flow(row['of_path'])
                        if self.config_dict['rgb_period'] > 1:
                            # We only want the first two channels of the flow.
                            flow = np.take(flow, [0,1], axis=2)
                        flow_seq_list.append(flow)

                    seq_index += 1

                if batch_index == 0:
                    X_batch_list = []
                    y_batch_list = []
                    flow_batch_list = []

                if seq_index == self.seq_length:
                    # *We only have per-clip labels, so the pain levels should not differ.
                    assert(len(set(y_seq_list)) == 1)
                    if self.config_dict['rgb_period'] > 1:
                        flow_seq_list = np.array(flow_seq_list)
                        flow_seq_list = np.reshape(np.array(flow_seq_list),
                                                  (-1, self.image_size[0], self.image_size[1]))
                        X_seq_list = np.reshape(np.array(X_seq_list),
                                               (self.image_size[0], self.image_size[1], -1))
                        
                    X_batch_list.append(X_seq_list)
                    y_batch_list.append(y_seq_list[0])  # *only need one
                    flow_batch_list.append(flow_seq_list)
                    batch_index += 1
                    seq_index = 0
                    
                    if train and (self.aug_flip == 1):
                        # Flip both RGB and flow arrays
                        X_seq_list_flipped = self.flip_images(X_seq_list)
                        flow_seq_list_flipped = self.flip_images(flow_seq_list)
                        # Append to the respective batch lists
                        X_batch_list.append(X_seq_list_flipped)
                        y_batch_list.append(y_seq_list[0])
                        flow_batch_list.append(flow_seq_list_flipped)
                        batch_index += 1

                    if train and (self.aug_crop == 1):
                        crop_size = 99
                        # Flip both RGB and flow arrays
                        X_seq_list_cropped = self.random_crop_resize(X_seq_list,
                                                                     crop_size, crop_size)
                        flow_seq_list_cropped = self.random_crop_resize(flow_seq_list,
                                                                        crop_size, crop_size)
                        # Append to the respective batch lists
                        X_batch_list.append(X_seq_list_cropped)
                        y_batch_list.append(y_seq_list[0])
                        flow_batch_list.append(flow_seq_list_cropped)
                        batch_index += 1

                    if train and (self.aug_light == 1):
                        # Flip both RGB and flow arrays
                        X_seq_list_shaded = self.add_gaussian_noise(X_seq_list)
                        flow_seq_list_shaded = self.add_gaussian_noise(flow_seq_list)
                        # Append to the respective batch lists
                        X_batch_list.append(X_seq_list_shaded)
                        y_batch_list.append(y_seq_list[0])
                        flow_batch_list.append(flow_seq_list_shaded)
                        batch_index += 1

                    # if train:
                    #     plot_augmentation(train, val, test, evaluate, 1, X_seq_list,
                    #         X_seq_list_flipped, X_seq_list_cropped, X_seq_list_shaded,
                    #         seq_index, batch_index, window_index)
                    #     plot_augmentation(train, val, test, evaluate, 0, flow_seq_list,
                    #         flow_seq_list_flipped, flow_seq_list_cropped, flow_seq_list_shaded,
                    #         seq_index, batch_index, window_index)

                if batch_index % self.batch_size == 0 and not batch_index == 0:
                    X_array = np.array(X_batch_list, dtype=np.float32)
                    y_array = np.array(y_batch_list, dtype=np.uint8)
                    flow_array = np.array(flow_batch_list, dtype=np.float32)
                    if self.nb_labels == 2:
                        y_array = tf.keras.utils.to_categorical(y_array, num_classes=self.nb_labels)
                    y_array = np.reshape(y_array, (self.batch_size, self.nb_labels))
                    batch_index = 0
                    yield [X_array, flow_array], y_array

    def prepare_image_generator_5D(self, df, train):
        """
        Prepare the frames into labeled train and test sets, with help from the
        DataFrame with .jpg-paths and labels for train and pain.
        :param df: pd.DataFrame
        :param train: Boolean
        :param config_dict: dict
        :return: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """
        nb_frames = len(df)
        print("LEN DF, in prep_5d(): ", nb_frames)

        window_size = self.config_dict['seq_length']
        window_stride = self.config_dict['seq_stride']
        last_valid_start_index = nb_frames - (window_size - 1)
        last_valid_end_index = last_valid_start_index + (window_size-1)
        number_of_windows = last_valid_end_index // window_stride

        assert (number_of_windows >= self.config_dict['batch_size'])
        print('Number of windows', number_of_windows)

        this_index = 0
        seq_index = 0
        
        # Make sure that no augmented sequences are thrown away,
        # because we really want to augment the dataset.

        nb_aug = self.aug_flip + self.aug_crop + self.aug_light
        batch_requirement = 1 + nb_aug  # Normal sequence plus augmented sequences.
        assert (self.batch_size % batch_requirement) == 0

        while True:
            # Shuffle blocks between epochs if during training.
            if train:
                df = shuffle_blocks(df, 'video_id')
            batch_index = 0
            for window_index in range(number_of_windows):
                start = window_index * window_stride
                stop = start + window_size
                rows = df.iloc[start:stop]  # A new dataframe for the window in question.

                X_seq_list = []
                y_seq_list = []

                for index, row in rows.iterrows():
                    vid_seq_name = row['video_id']

                    if this_index == 0:
                        old_vid_seq_name = vid_seq_name # Set this variable (only once).
                        this_index += 1

                    if vid_seq_name != old_vid_seq_name:
                        seq_index = 0
                        old_vid_seq_name = vid_seq_name
                        break  # In that case want to jump to the next window.

                    if self.config_dict['data_type'] == 'rgb':
                        x = self.get_image(row['path'])
                    if self.config_dict['data_type'] == 'of':
                        x = self.get_flow(row['of_path'])
                        # If no magnitude:
                        # extra_channel = np.zeros((x.shape[0], x.shape[1], 1))
                        # x = np.concatenate((x, extra_channel), axis=2)
                    y = row['pain']
                    X_seq_list.append(x)
                    y_seq_list.append(y)
                    seq_index += 1

                if batch_index == 0:
                    X_batch_list = []
                    y_batch_list = []

                if seq_index == self.seq_length:
                    X_batch_list.append(X_seq_list)
                    y_batch_list.append(y_seq_list[0])
                    seq_index = 0
                    batch_index += 1

                    if train and (self.aug_flip == 1):
                        X_seq_list_flipped = self.flip_images(X_seq_list)
                        X_batch_list.append(X_seq_list_flipped)
                        y_batch_list.append(y_seq_list[0])
                        batch_index += 1

                    if train and (self.aug_crop == 1):
                        crop_size = 99
                        X_seq_list_cropped = self.random_crop_resize(X_seq_list,
                                                                     crop_size, crop_size)
                        X_batch_list.append(X_seq_list_cropped)
                        y_batch_list.append(y_seq_list[0])
                        batch_index += 1

                    if train and (self.aug_light == 1):
                        X_seq_list_shaded = self.add_gaussian_noise(X_seq_list)
                        X_batch_list.append(X_seq_list_shaded)
                        y_batch_list.append(y_seq_list[0])
                        batch_index += 1

                    # if train:
                    #     plot_augmentation(train, val, test, evaluate, 1, X_seq_list,
                    #         X_seq_list_flipped, X_seq_list_cropped, X_seq_list_shaded,
                    #         seq_index, batch_index, window_index)

                if batch_index % self.batch_size == 0 and not batch_index == 0:
                    X_array = np.array(X_batch_list, dtype=np.float32)
                    y_array = np.array(y_batch_list, dtype=np.uint8)
                    if self.nb_labels == 2:
                        y_array = tf.keras.utils.to_categorical(y_array, num_classes=self.nb_labels)
                    y_array = np.reshape(y_array, (self.batch_size, self.nb_labels))
                    batch_index = 0
                    yield X_array, y_array

    def prepare_image_generator(self, df, train):
        """
        Prepare the frames into labeled train and test sets, with help from the
        DataFrame with .jpg-paths and labels for train and pain.
        :param df: pd.DataFrame
        :param train: Boolean
        :param config_dict: dict
        :return: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """
        print("LEN DF:")
        print(len(df))
        print('Datatype:')
        print(self.config_dict['data_type'])

        while True:
            if train:
                # Shuffle blocks between epochs.
                df = shuffle_blocks(df, 'video_id')
            batch_index = 0
            for index, row in df.iterrows():
                if batch_index == 0:
                    X_list = []
                    y_list = []
                if self.config_dict['data_type'] == 'rgb':
                    x = self.get_image(row['path'])
                    x /= 255
                if self.config_dict['data_type'] == 'of':
                    x = self.get_flow(row['of_path'])
                y = row['pain']
                X_list.append(x)
                y_list.append(y)
                batch_index += 1

                if batch_index % self.batch_size == 0:
                    X_array = np.array(X_list, dtype=np.float32)
                    y_array = np.array(y_list, dtype=np.uint8)
                    y_array = tf.keras.utils.to_categorical(y_array,
                                                      num_classes=self.nb_labels)
                    batch_index = 0
                    yield (X_array, y_array)

    def get_image(self, path):
        im = helpers.process_image(
            path, (self.image_size[0], self.image_size[1], self.color_channels),
            standardize=True, mean=self.pixel_mean, std=self.pixel_std)
        return im

    def get_flow(self, path):
        flow = helpers.process_image(
            path, (self.image_size[0], self.image_size[1], self.color_channels),
            standardize=False)
        return flow

    def flip_images(self, images):
        X_flip = []
        tf.reset_default_graph()
        # Tensorflow wants [height, width, channels] input below, hence [1] before [0].
        X = tf.placeholder(tf.float32, shape=(self.image_size[1], self.image_size[0], 3))
        tf_img1 = tf.image.flip_left_right(X)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for img in images:
                flipped_imgs = sess.run([tf_img1], feed_dict={X: img})
                X_flip.extend(flipped_imgs)
        X_flip = np.array(X_flip, dtype=np.float32)
        return X_flip

    def flip_image(self, image):
        tf.reset_default_graph()
        # Tensorflow wants [height, width, channels] input below, hence [1] before [0].
        X = tf.placeholder(tf.float32, shape=(self.image_size[1], self.image_size[0], 3))
        tf_img1 = tf.image.flip_left_right(X)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            X_flip = sess.run([tf_img1], feed_dict={X:image})
        X_flip = np.array(X_flip, dtype=np.float32)
        X_flip = np.reshape(X_flip, (self.image_size[1], self.image_size[0], 3))
        return X_flip

    def add_gaussian_noise(self, images):
        """
        This methods shadens the images with Gaussian noise.
        """
        gaussian_noise_imgs = []

        row, col = self.image_size
    
        mean = 0
        sigma = 0.5

        imw_a = 0.55
        imw_b = 0.7
        im_weight = (imw_b - imw_a) * np.random.random() + imw_a

        now_a = 0.2
        now_b = 0.4
        noise_weight = (now_b - now_a) * np.random.random() + now_a

        gaussian = np.random.normal(mean, sigma, (col, row, self.color_channels)).astype(np.float32)

        if self.nb_input_dims == 5:
            for img in images:
                gaussian_img = cv2.addWeighted(img, im_weight, gaussian, noise_weight, 0)
                gaussian_noise_imgs.append(gaussian_img)
        if self.nb_input_dims == 4:
            gaussian_noise_imgs = cv2.addWeighted(images, im_weight, gaussian, noise_weight, 0)
            # gaussian_noise_imgs.append(gaussian_img)
    
        gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.float32)
        return gaussian_noise_imgs

    def add_gaussian_noise_to_single_image(self, image):
        """
        This methods shadens the images with Gaussian noise.
        """
        row, col = self.image_size
    
        mean = 0
        sigma = 0.5

        imw_a = 0.55
        imw_b = 0.7
        im_weight = (imw_b - imw_a) * np.random.random() + imw_a

        now_a = 0.2
        now_b = 0.4
        noise_weight = (now_b - now_a) * np.random.random() + now_a

        gaussian = np.random.normal(mean, sigma, (col, row, self.color_channels)).astype(np.float32)

        gaussian_noise_img = cv2.addWeighted(image, im_weight, gaussian, noise_weight, 0)
    
        gaussian_noise_img = np.array(gaussian_noise_img, dtype=np.float32)
        return gaussian_noise_img

    def random_crop_resize(self, images, target_height, target_width):
        """
        Random crop but consistent across sequence.
        :param images:
        :param target_height:
        :param target_width:
        :return:
        """
        random_scale_for_crop_w = np.random.rand()
        random_scale_for_crop_h = np.random.rand()

        crop_scale_w = random_scale_for_crop_w * 0.2
        crop_scale_h = random_scale_for_crop_h * 0.2
        # print('Crop scale w and h: ', crop_scale_w, crop_scale_h)

        width = self.image_size[0]
        height = self.image_size[1]
        offset_height = crop_scale_h * height
        offset_width = crop_scale_w * width

        # y1 x1 are relative starting heights and widths in the crop box.
        # [[0, 0, 1, 1]] would mean no crop and just resize.
    
        y1 = offset_height/(height-1)
        x1 = offset_width/(width-1)
        y2 = (offset_height + target_height)/(height-1)
        x2 = (offset_width + target_width)/(width-1)
    
        boxes = np.array([[y1, x1, y2, x2]], dtype=np.float32)
        box_ind = np.array([0], dtype=np.int32)
        crop_size = np.array([width, height], dtype=np.int32)
    
        X_crops = []
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, shape=(1, height, width, 3))
        tf_img1 = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if self.nb_input_dims == 5:
                for img in images:
                    batch_img = np.expand_dims(img, axis = 0)
                    cropped_imgs = sess.run([tf_img1], feed_dict={X: batch_img})
                    X_crops.extend(cropped_imgs)
            if self.nb_input_dims == 4:
                batch_img = np.expand_dims(images, axis = 0)
                cropped_imgs = sess.run([tf_img1], feed_dict={X: batch_img})
                X_crops.extend(cropped_imgs)
        X_crops = np.array(X_crops, dtype=np.float32)
        if self.nb_input_dims == 5:
            X_crops = np.reshape(X_crops, (self.seq_length, height, width, 3))
        if self.nb_input_dims == 4:
            X_crops = np.reshape(X_crops, (height, width, 3))
        return X_crops

    def random_crop_resize_single_image(self, image, target_height, target_width):
        """
        Random crop but consistent across sequence.
        :param images:
        :param target_height:
        :param target_width:
        :return:
        """
        random_scale_for_crop_w = np.random.rand()
        random_scale_for_crop_h = np.random.rand()

        crop_scale_w = random_scale_for_crop_w * 0.2
        crop_scale_h = random_scale_for_crop_h * 0.2
        # print('Crop scale w and h: ', crop_scale_w, crop_scale_h)

        width = self.image_size[0]
        height = self.image_size[1]
        offset_height = crop_scale_h * height
        offset_width = crop_scale_w * width

        # y1 x1 are relative starting heights and widths in the crop box.
        # [[0, 0, 1, 1]] would mean no crop and just resize.
    
        y1 = offset_height/(height-1)
        x1 = offset_width/(width-1)
        y2 = (offset_height + target_height)/(height-1)
        x2 = (offset_width + target_width)/(width-1)
    
        boxes = np.array([[y1, x1, y2, x2]], dtype=np.float32)
        box_ind = np.array([0], dtype=np.int32)
        crop_size = np.array([width, height], dtype=np.int32)
    
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, shape=(1, height, width, 3))
        tf_img1 = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batch_img = np.expand_dims(image, axis = 0)
            cropped_img = sess.run([tf_img1], feed_dict={X: batch_img})
        X_crops = np.array(cropped_img, dtype=np.float32)
        X_crops = np.reshape(X_crops, (height, width, 3))
        return X_crops

    def subject_to_df(self, subject_id, dataset, config_file):
        """
        Create a DataFrame with all the frames with annotations from a csv-file.
        :param subject_id: int
        :param dataset: str
        :param config_file: dict
        :return: pd.DataFrame
        """
        clip_file = config_file['clip_list_pf']\
            if dataset == 'pf' else config_file['clip_list_lps']
        df_csv = pd.read_csv(clip_file)
        column_headers = ['video_id', 'path', 'train']
        for dc in self.data_columns:
            column_headers.append(dc)
        print(column_headers)
        subject_path = os.path.join(
            self.dataset_rgb_path_dict[dataset], subject_id)
        big_list = []
        for path, dirs, files in sorted(os.walk(subject_path)):
            print(path)
            for filename in sorted(files):
                # if '.jpg' in filename or '.png' in filename:
                if filename.startswith('frame_')\
                        and ('.jpg' in filename or '.png' in filename):
                    total_path = os.path.join(path, filename)
                    print(total_path)
                    vid_id = get_video_id_stem_from_path(path)
                    csv_row = df_csv.loc[df_csv['video_id'] == vid_id]
                    if csv_row.empty:
                        continue
                    train_field = -1
                    row_list = [vid_id, total_path, train_field]
                    for dc in self.data_columns:
                        field = csv_row.iloc[0][dc]
                        if dc == 'pain':
                            pain = 1 if field > 0 else 0
                            field = pain
                        row_list.append(field)
                    big_list.append(row_list)
        subject_df = pd.DataFrame(big_list, columns=column_headers)
        return subject_df

    def save_OF_paths_to_df(self, subject_id, subject_df, dataset):
        """
        Create a DataFrame with all the optical flow paths with annotations
        from a csv-file, then join it with the existing subject df with rgb paths,
        at simultaneous frames.
        :param subject_id: int
        :param subject_df: pd.DataFrame
        :param dataset: str
        :return: pd.DataFrame
        """
        c = 0  # Per subject frame counter.
        per_clip_frame_counter = 0
        old_path = 'NoPath'
        subject_path = os.path.join(
            self.dataset_of_path_dict[dataset], subject_id)
        of_path_list = []
        list_of_video_ids = list(set(subject_df['video_id'].values))

        # Walk through all the files in the of-folders and put them in a
        # list, in order (the same order they were extracted in.)

        for path, dirs, files in sorted(os.walk(subject_path)):
            print(path)
            video_id = get_video_id_from_path(path)
            if video_id not in list_of_video_ids:
                print(video_id, ' was excluded')
                continue
            print(video_id)
            nb_frames_in_clip = len(
                subject_df.loc[subject_df['video_id'] == video_id])
            if old_path != path and c != 0:  # If entering a new folder (but not first time)
                per_clip_frame_counter = 0
                if '1fps' in subject_path or 'ShoulderPain' in subject_path:
                    print('Dropping first optical flow to match with rgb.')
                    subject_df.drop(c, inplace=True)  # Delete first element
                    subject_df.reset_index(drop=True, inplace=True)  # And adjust the index
            old_path = path
            for filename in sorted(files):
                total_path = os.path.join(path, filename)
                if filename.startswith('flow_')\
                        and ('.npy' in filename or '.jpg' in filename): 
                    of_path_list.append(total_path)
                    c += 1
                    per_clip_frame_counter += 1
            if per_clip_frame_counter < nb_frames_in_clip:
                diff = nb_frames_in_clip - per_clip_frame_counter
                if diff > 1:
                    print('Warning: flow/rgb diff is larger than 1')
                else:
                    print('Counted {} flow-frames for {} rgb frames \n'.format(
                            per_clip_frame_counter, nb_frames_in_clip))
                    subject_df.drop(c, inplace=True)
                    subject_df.reset_index(drop=True, inplace=True)
                    print('Dropped the last rgb frame of the clip. \n')

        # Now extend subject_df to contain both rgb and OF paths,
        # and then return whole thing.
        try:
            subject_df.loc[:, 'of_path'] = pd.Series(of_path_list)
            subject_df.loc[:, 'train'] = -1
        except AssertionError:
            print('RGB and flow columns were not the same length'
                  'and the data could not be merged.')

        return subject_df


def plot_augmentation(train, val, test, rgb, X_seq_list, flipped, cropped, shaded,
                      seq_index, batch_index, window_index):
    rows = 4
    cols = 10
    f, axarr = plt.subplots(rows, cols, figsize=(20,10))
    for i in range(0, rows):
        for j in range(0, cols):
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
            if i == 0:
                im = X_seq_list[j]
                im /= 255
                axarr[i, j].imshow(im)
            elif i == 1:
                im = flipped[j]
                im /= 255
                axarr[i, j].imshow(im)
            elif i == 2:
                im = cropped[j]
                im /= 255
                axarr[i, j].imshow(im)
            else:
                im = shaded[j]
                im /= 255
                axarr[i, j].imshow(im)
    plt.tick_params(axis='both', which='both', bottom='off', left='off')
    f.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    if train:
        partition = 1
    elif val:
        partition = 2
    elif test:
        partition = 3
    else:
        partition = 4
    plt.savefig('seq_{}_batch_{}_wi_{}_part_{}_rgb_{}.png'.format(
        seq_index,
        batch_index,
        window_index,
        partition,
        rgb))
    plt.close()


def get_video_id_stem_from_path(path):
    _, vid_id = helpers.split_string_at_last_occurence_of_certain_char(path, '/')
    return vid_id


def get_video_id_from_path(path):
    _, vid_id = helpers.split_string_at_last_occurence_of_certain_char(path, '/')
    return vid_id


def get_video_id_from_frame_path(path):
    path_left, frame_id = helpers.split_string_at_last_occurence_of_certain_char(path, '/')
    _, vid_id = helpers.split_string_at_last_occurence_of_certain_char(path_left, '/')
    return vid_id


def make_even_sequences(x, seq_length):
    x = round_to_batch_size(np.asarray(x, dtype=np.float32), seq_length)
    num_splits = int(float(len(x)) / seq_length)
    x = np.split(np.asarray(x, dtype=np.float32), num_splits)
    return np.asarray(x)


def round_to_batch_size(data_array, batch_size):
    num_rows = data_array.shape[0]
    surplus = num_rows % batch_size
    data_array_rounded = data_array[:num_rows-surplus]
    return data_array_rounded


def shuffle_blocks(df, key):
    """
    Takes a dataframe with all frames from all different sequences (which always
    lay in the same order) and shuffles the blocks of frames from separate videos,
    without altering the internal frame-ordering.
    The intention is that
    :param df: pd.Dataframe
    :param key: str
    :return:
    """
    vids = set(df[key])
    df_blocks = []
    for v in vids:
        df_block = df[df[key] == v]
        df_blocks.append(df_block)
    random.shuffle(df_blocks)
    df = pd.concat(df_blocks)
    return df


def get_flow_magnitude(flow):
    """
    Compute the magnitude of the optical flow at every pixel.
    :param flow: np.ndarray [width, height, 2]
    :return: np.ndarray [width, height, 1]
    """
    rows = flow.shape[0]
    cols = flow.shape[1]
    magnitude = np.zeros((rows, cols, 1))
    for i in range(0, rows):
        for j in range(0, cols):
            xflow = flow[i, j, 0]
            yflow = flow[i, j, 1]
            mag = np.sqrt(np.power(xflow, 2) + np.power(yflow, 2))
            magnitude[i, j] = mag
    return magnitude

