import tensorflow as tf
import pandas as pd
import numpy as np
import helpers
import random
import cv2
import re
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class DataHandler:
    def __init__(self, data_columns, config_dict, all_subjects_df):
        """
        :param data_columns: [str]
        :param config_dict: dict
        :param all_subjects_df: pd.DataFrame
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

    def get_dataset(self, sequence_dfs, train):
        """
        From frame paths to tf.data.Dataset consisting of sequences.
        :param sequence_dfs: [pd.DataFrame]
        :param train:  boolean
        :return: tf.data.Dataset
        """

        if self.config_dict['nb_input_dims'] == 5:
            if '2stream' in self.config_dict['model']:
                if self.config_dict['save_features']:
                    dataset = tf.data.Dataset.from_generator(
                        lambda: self.prepare_2stream_image_generator_5D_with_paths(sequence_dfs, train),
                        output_types=(tf.float32, tf.uint8, tf.string))
                else:
                    dataset = tf.data.Dataset.from_generator(
                        lambda: self.prepare_2stream_image_generator_5D(sequence_dfs, train),
                        output_types=(tf.float32, tf.uint8))
            else:
                dataset = tf.data.Dataset.from_generator(
                    lambda: self.prepare_image_generator_5D(sequence_dfs, train),
                    output_types=(tf.float32, tf.uint8),
                    output_shapes=(
                        tf.TensorShape([None, self.config_dict['seq_length'], self.image_size[0],
                                        self.image_size[1], self.color_channels]),
                        tf.TensorShape([None, 2]))
                )
        if self.config_dict['nb_input_dims'] == 4:
            if '2stream' in self.config_dict['model']:
                generator = self.prepare_generator_2stream(
                    sequence_dfs, train
                )
            else:
                generator = self.prepare_image_generator(
                    sequence_dfs, train
                )

        return dataset

    def features_to_dataset(self, subjects):
        subj_codes = []
        for subj in subjects:
            code = self.all_subjects_df[
                self.all_subjects_df['subject'] == subj]['code']
            subj_codes.append(code.values[0])
        dataset = tf.data.Dataset.from_generator(
            lambda: self.generate_features(subject_codes=subj_codes,
                                           ),
            output_types=(tf.float32, tf.float32, tf.uint8, tf.string),
            output_shapes=(tf.TensorShape([None, None]),
                           tf.TensorShape([None, 2]),
                           tf.TensorShape([None, 2]),
                           tf.TensorShape([]))
        )
        print('Shuffling dataset...')
        dataset = dataset.shuffle(
            self.config_dict['shuffle_buffer'], reshuffle_each_iteration=True)
        dataset = dataset.padded_batch(self.config_dict['video_batch_size'])
        return dataset

    def generate_features(self,
                          subject_codes,
                          features_folder='video_level_features_320dim/'):
        """
        Load features from file (per video).
        :param subject_codes: [str]
        :param features_folder: str
        :yield: batch features, batch preds, batch labels
        """
        path_to_features = self.config_dict['data_path'] + 'lps/' + features_folder
        df_summary = pd.read_csv(path_to_features + 'summary.csv')
        subj_dfs = []
        for subj_code in subject_codes:
            subject_selected_df = df_summary[(df_summary.subject == subj_code)]
            subj_dfs.append(subject_selected_df)

        df = pd.concat(subj_dfs)

        default_array_str = 'arr_0'

        for index, row in df.iterrows():
            video_id = row['video_id']
            path = path_to_features + video_id + '.npz'
            loaded = np.load(path, allow_pickle=True)[default_array_str].tolist()
            feats = loaded['features']
            f_shape = feats.shape
            preds = np.array(loaded['preds'])
            labels = np.array(loaded['labels'])
            assert preds.shape[0] == f_shape[0]
            assert labels.shape[0] == f_shape[0]
            yield feats, preds, labels, video_id

    def prepare_video_features(self, features):
        save_folder = 'data/lps/video_level_features_320dim/'
        default_array_str = 'arr_0'
        nb_clip_batches = features[default_array_str].shape[0]
        col_headers = ['subject', 'video_id', 'length']
        big_list = []
        for clip_batch in range(nb_clip_batches):
            clip_batch_feats = features[default_array_str][clip_batch]['features'].numpy()
            clip_batch_preds = features[default_array_str][clip_batch]['preds'].numpy()
            clip_batch_labels = features[default_array_str][clip_batch]['y'].numpy()
            clip_batch_paths = features[default_array_str][clip_batch]['paths'].numpy()
            for ind, path in enumerate(clip_batch_paths):
                video_id = get_video_id_from_frame_path(str(path))
                if clip_batch == 0 and ind == 0:
                    print('start, video id: ', video_id)
                    old_video_id = video_id
                    same_video_features = []
                    same_video_preds = []
                    same_video_labels = []

                if video_id != old_video_id:
                    print('new video id: ', video_id)
                    old_video_id = video_id

                    feats = np.array(same_video_features)
                    f_shape = feats.shape
                    feats = np.reshape(
                        # feats, [f_shape[0], f_shape[1] * f_shape[2] * f_shape[3] * f_shape[4]])
                        feats, [f_shape[0], f_shape[1] * f_shape[2]])
                    preds = same_video_preds
                    labels = same_video_labels
                    to_save_dict = {}
                    to_save_dict['features'] = feats
                    to_save_dict['preds'] = preds
                    to_save_dict['labels'] = labels
                    length = f_shape[0]
                    to_save_dict = np.array(to_save_dict)
                    save_filename = save_folder + video_id + '.npz'
                    subject = video_id[0]
                    video_list = [subject, video_id, length]
                    big_list.append(video_list)
                    np.savez_compressed(save_filename, to_save_dict)
                    same_video_features = []
                    same_video_preds = []
                    same_video_labels = []

                same_video_features.append(clip_batch_feats[ind])
                same_video_preds.append(clip_batch_preds[ind])
                same_video_labels.append(clip_batch_labels[ind])

        # Finally also use the last one (video ID doesn't change)

        feats = np.array(same_video_features)
        f_shape = feats.shape
        feats = np.reshape(
            # feats, [f_shape[0], f_shape[1] * f_shape[2] * f_shape[3] * f_shape[4]])
            feats, [f_shape[0], f_shape[1] * f_shape[2]])
        to_save_dict = {}
        to_save_dict['features'] = feats
        to_save_dict['preds'] = preds
        to_save_dict['labels'] = labels
        subject = video_id[0]
        length = f_shape[0]
        video_list = [subject, video_id, length]
        big_list.append(video_list)
        video_df = pd.DataFrame(big_list, columns=col_headers)
        save_filename = save_folder + video_id + '.npz'
        np.savez_compressed(save_filename, to_save_dict)
        video_df.to_csv(save_folder + 'summary.csv')

    def generate_video_features(self, features):
        default_array_str = 'arr_0'
        nb_clip_batches = features[default_array_str].shape[0]
        for clip_batch in range(nb_clip_batches):
            clip_batch_feats = features[default_array_str][clip_batch]['features'].numpy()
            clip_batch_preds = features[default_array_str][clip_batch]['preds'].numpy()
            clip_batch_labels = features[default_array_str][clip_batch]['y'].numpy()
            clip_batch_paths = features[default_array_str][clip_batch]['paths'].numpy()
            for ind, path in enumerate(clip_batch_paths):
                video_id = get_video_id_from_frame_path(str(path))
                if clip_batch == 0 and ind == 0:
                    print('start, video id: ', video_id)
                    old_video_id = video_id
                    same_video_features = []
                    same_video_preds = []
                    same_video_labels = []

                if video_id != old_video_id:
                    print('new video id: ', video_id)
                    old_video_id = video_id

                    feats = np.array(same_video_features)
                    f_shape = feats.shape
                    feats = np.reshape(
                        # feats, [f_shape[0], f_shape[1] * f_shape[2] * f_shape[3] * f_shape[4]])
                        feats, [f_shape[0], f_shape[1] * f_shape[2]])
                    preds = same_video_preds
                    labels = same_video_labels
                    same_video_features = []
                    same_video_preds = []
                    same_video_labels = []

                    yield feats, preds, labels

                same_video_features.append(clip_batch_feats[ind])
                same_video_preds.append(clip_batch_preds[ind])
                same_video_labels.append(clip_batch_labels[ind])

        # Finally also yield the last one.

        feats = np.array(same_video_features)
        f_shape = feats.shape
        feats = np.reshape(
            # feats, [f_shape[0], f_shape[1] * f_shape[2] * f_shape[3] * f_shape[4]])
            feats, [f_shape[0], f_shape[1] * f_shape[2]])
        yield feats, same_video_preds, same_video_labels

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
        df = shuffle_blocks(df, 'video_id')
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

    def set_train_val_test_in_df(self, dfs):
        """
        Mark in input dataframe which subjects to use for train, val or test.
        Used when val_mode == 'subject'
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
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        if not self.config_dict['val_mode'] == 'no_val':
            df_val.reset_index(drop=True, inplace=True)
        else:
            df_val = []

        train_sequence_dfs = self.get_sequences_from_frame_df(df=df_train)
        test_sequence_dfs = self.get_sequences_from_frame_df(df=df_test)

        if not self.config_dict['val_mode'] == 'no_val':
            val_sequence_dfs = self.get_sequences_from_frame_df(df=df_val)
        else:
            val_sequence_dfs = []

        print("\nNb. of train, val and test frames: ",
              len(df_train), len(df_val), len(df_test), '\n')

        print("...resulting in nb. of train, val and test sequences: ",
              len(train_sequence_dfs),
              len(val_sequence_dfs),
              len(test_sequence_dfs), '\n')

        return train_sequence_dfs, val_sequence_dfs, test_sequence_dfs

    def get_datasets(self, df_train, df_val, df_test):
        train_dataset = self.get_dataset(df_train, train=True)
        val_dataset = self.get_dataset(df_val, train=False)
        test_dataset = self.get_dataset(df_test, train=False)

        return train_dataset, val_dataset, test_dataset

    def prepare_generator_2stream(self, df, train):
        """
        Prepare batches of frames, optical flow, and labels,
        with help from the DataFrame with frame paths and labels.
        :param df: pd.DataFrame
        :param train: Boolean
        :return: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """
        print("LEN DF (nb. of frames): ", len(df))

        # Make sure that no augmented batches are thrown away,
        # because we really want to augment the dataset.

        nb_aug = self.aug_flip + self.aug_crop + self.aug_light
        batch_requirement = 1 + nb_aug  # Normal sequence plus augmented sequences.
        assert (self.batch_size % batch_requirement) == 0
        while True:
            # Shuffle videos between epochs.
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
                yield [X_array, flow_array], y_array

    def prepare_2stream_image_generator_5D(self, sequence_dfs, train):
        """
        Prepare batches of frame sequences, optical flow sequences,
        and labels, with help from the DataFrame with frame paths and labels.
        :param sequence_dfs: pd.DataFrame
        :param train: Boolean
        :return: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """
        nb_aug = self.aug_flip + self.aug_crop + self.aug_light
        batch_requirement = 1 + nb_aug  # Normal sequence plus augmented sequences.
        assert (self.batch_size % batch_requirement) == 0

        while True:
            if train:  # Shuffle videos between epochs.
                print('Shuffling the order of sequences.')
                random.shuffle(sequence_dfs)

            batch_index = 0
            for sequence_df in sequence_dfs:

                X_seq_list = []
                y_seq_list = []
                flow_seq_list = []

                for seq_index, row in sequence_df.iterrows():

                    if (seq_index % self.config_dict['rgb_period']) == 0:
                        x = self.get_image(row['path'])
                        y = row['pain']
                        X_seq_list.append(x)
                        y_seq_list.append(y)

                    if (seq_index % self.config_dict['flow_period']) == 0:
                        flow = self.get_flow(row['of_path'])
                        if self.config_dict['rgb_period'] > 1:
                            # We only want the first two channels of the flow
                            flow = np.take(flow, [0, 1], axis=2)  # Simonyan type input
                        flow_seq_list.append(flow)

                if batch_index == 0:
                    X_batch_list = []
                    y_batch_list = []
                    flow_batch_list = []

                # *We only have per-clip labels, so the pain levels should not differ.
                assert (len(set(y_seq_list)) == 1)
                if self.config_dict['rgb_period'] > 1:
                    flow_seq_list = np.array(flow_seq_list)
                    flow_seq_list = np.reshape(np.array(flow_seq_list),
                                               (-1, self.image_size[0], self.image_size[1]))
                    X_seq_list = np.reshape(np.array(X_seq_list),
                                            (self.image_size[0], self.image_size[1], -1))

                X_batch_list.append(X_seq_list)
                flow_batch_list.append(flow_seq_list)
                y_batch_list.append(y_seq_list[0])  # *only need one
                batch_index += 1

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

    def prepare_2stream_image_generator_5D_with_paths(self, sequence_dfs, train):
        """
        Prepare batches of frame sequences, optical flow sequences,
        and labels, with help from the DataFrame with frame paths and labels.
        :param sequence_dfs: pd.DataFrame
        :param train: Boolean
        :return: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """
        while True:
            batch_index = 0
            for sequence_df in sequence_dfs:

                X_seq_list = []
                y_seq_list = []
                flow_seq_list = []
                path_seq_list = []

                for seq_index, row in sequence_df.iterrows():

                    if (seq_index % self.config_dict['rgb_period']) == 0:
                        x = self.get_image(row['path'])
                        y = row['pain']
                        X_seq_list.append(x)
                        y_seq_list.append(y)
                        path_seq_list.append(row['path'])  # Only save one (last) path per seq

                    if (seq_index % self.config_dict['flow_period']) == 0:
                        flow = self.get_flow(row['of_path'])
                        if self.config_dict['rgb_period'] > 1:
                            # We only want the first two channels of the flow
                            flow = np.take(flow, [0, 1], axis=2)  # Simonyan type input
                        flow_seq_list.append(flow)

                if batch_index == 0:
                    X_batch_list = []
                    y_batch_list = []
                    flow_batch_list = []
                    path_batch_list = []

                # *We only have per-clip labels, so the pain levels should not differ.
                assert (len(set(y_seq_list)) == 1)

                X_batch_list.append(X_seq_list)
                flow_batch_list.append(flow_seq_list)
                y_batch_list.append(y_seq_list[0])  # *only need one
                path_batch_list.append(path_seq_list[0])  # *only need one
                batch_index += 1

                if batch_index % self.batch_size == 0 and not batch_index == 0:
                    X_array = np.array(X_batch_list, dtype=np.float32)
                    y_array = np.array(y_batch_list, dtype=np.uint8)
                    flow_array = np.array(flow_batch_list, dtype=np.float32)
                    path_array = np.array(path_batch_list)
                    if self.nb_labels == 2:
                        y_array = tf.keras.utils.to_categorical(y_array, num_classes=self.nb_labels)
                    y_array = np.reshape(y_array, (self.batch_size, self.nb_labels))
                    batch_index = 0
                    yield [X_array, flow_array], y_array, path_array

    def get_sequences_from_frame_df(self, df):
        """
        Given a dataframe of all frame paths, video IDs and labels,
        and some sequence length and stride, return a list of
        [sequence length]-long dataframes to use for reading in data.
        :param df: pd.DataFrame
        :return: [pd.DataFrame]
        """
        # print('\nPreparing sequences from list of frames...')
        # nb_frames = len(df)
        # print('Number of frames in df: ', nb_frames)

        def build_sequences_from_frames(start_ind, video_frame_df, nb_per_video=None):
            nb_frames_in_video = len(video_frame_df)
            sequence_dfs_from_one_video = []

            window_size = self.config_dict['seq_length']
            window_stride = self.config_dict['seq_stride']
            last_valid_start_index = nb_frames_in_video - window_size
            last_valid_end_index = nb_frames_in_video - 1

            if nb_frames_in_video == self.config_dict['seq_length']:
                start_indices = [0]

            elif nb_per_video is None:
                number_of_windows = last_valid_end_index // window_stride
                # print('Number of windows', number_of_windows)
                start_indices = [(start_ind + window_index * window_stride)
                                 for window_index in range(number_of_windows)]
            else:  # Resampling for minor class
                # print('\nComputing start indices for resampling from the following df...\n')
                # print('video_frame_df.head():')
                # print(video_frame_df.head(), '\n')
                # print('Frames in video: {}, nb per video: {}, last valid start {}'.format(
                #     nb_frames_in_video, nb_per_video, last_valid_start_index))
                step_length = int((last_valid_start_index - start_ind)/nb_per_video)
                step_length = 1 if step_length == 0 else step_length
                approx_start_indices = [*range(start_ind, last_valid_start_index, step_length)]

                if step_length < self.config_dict['seq_stride']:
                    print('There might be repeated samples in the minor class.')
                    start_indices = approx_start_indices
                else:  # Choose to sample maximally off from first round
                    start_indices = []
                    for asi in approx_start_indices:
                        # Aim for nearest number X (to asi)
                        # where X % window_stride == start_ind
                        # but not X % window_length == 0
                        current_window_modulo = asi % window_size

                        if current_window_modulo == start_ind:
                            new_start_index = asi
                        elif current_window_modulo <= start_ind:
                            new_start_index = asi + (start_ind-current_window_modulo)
                        else:
                            new_start_index = asi - (current_window_modulo-start_ind)

                        assert(new_start_index % start_ind == 0)
                        assert(new_start_index % window_size == start_ind)

                        if new_start_index > last_valid_start_index:
                            break
                        else:
                            start_indices.append(new_start_index)
                # print(approx_start_indices)
                # print(start_indices)
                # print('\n')

            for start in start_indices:
                stop = start + window_size
                sequence_df = video_frame_df.iloc[start:stop]
                assert(len(sequence_df) == self.config_dict['seq_length'])
                sequence_dfs_from_one_video.append(sequence_df)

            return sequence_dfs_from_one_video

        def get_sequence_dfs_per_class(class_df, video_ids, start_ind):
            sequence_dfs_per_class = []
            if len(class_df) != 0:
                for video_id in video_ids:
                    video_frame_df = class_df.loc[class_df['video_id'] == video_id]
                    sequence_dfs_from_video = build_sequences_from_frames(
                        start_ind=start_ind,
                        video_frame_df=video_frame_df)
                    sequence_dfs_per_class += sequence_dfs_from_video
            return sequence_dfs_per_class

        def get_extra_sequences(class_df, video_ids, start_ind, nb_extra):
            sequence_dfs = []
            if len(class_df) != 0:
                if nb_extra < len(video_ids):
                    nb_per_video_to_sample = 1
                else:
                    nb_per_video_to_sample = int(nb_extra/len(video_ids))
                nb_sequences_collected = 0
                for video_id in video_ids:
                    video_frame_df = class_df.loc[class_df['video_id'] == video_id]
                    sequence_dfs_from_video = build_sequences_from_frames(
                        start_ind=start_ind,
                        video_frame_df=video_frame_df,
                        nb_per_video=nb_per_video_to_sample)
                    for seq_df in sequence_dfs_from_video:
                        assert(len(seq_df) == self.config_dict['seq_length'])
                        sequence_dfs.append(seq_df)
                        nb_sequences_collected += 1
                        if nb_sequences_collected == nb_extra:
                            break
            return sequence_dfs

        def get_class_dfs_and_video_ids():

            nopain_df = df.loc[df['pain'] == 0]
            nopain_video_ids = set(nopain_df['video_id'])

            pain_df = df.loc[df['pain'] == 1]
            pain_video_ids = set(pain_df['video_id'])
            return {'no_pain': (nopain_df, nopain_video_ids),
                    'pain': (pain_df, pain_video_ids)}

        class_dfs_dict = get_class_dfs_and_video_ids()

        print('Nb. videos for no pain: {}, nb. videos for pain: {}'.format(
            len(class_dfs_dict['no_pain'][1]),
            len(class_dfs_dict['pain'][1]),
        ))

        no_pain_sequence_dfs = get_sequence_dfs_per_class(
            class_df=class_dfs_dict['no_pain'][0],
            video_ids=class_dfs_dict['no_pain'][1],
            start_ind=0)
        pain_sequence_dfs = get_sequence_dfs_per_class(
            class_df=class_dfs_dict['pain'][0],
            video_ids=class_dfs_dict['pain'][1],
            start_ind=0)

        diff = len(no_pain_sequence_dfs) - len(pain_sequence_dfs)

        print('Diff: {}, nb. no pain sequences: {}, nb. pain sequences: {}'.format(
            diff, len(no_pain_sequence_dfs), len(pain_sequence_dfs)
        ))

        minor_class = 'pain' if diff > 0 else 'no_pain'
        resample_start_ind = int(
            self.config_dict['resample_start_fraction_of_seq_length']
            * self.config_dict['seq_length'])
        if abs(diff) > 0:
            print('Resampling from the {}th index within a window...'.format(
                resample_start_ind))
            extra_seqs_for_minor_class = get_extra_sequences(
                class_df=class_dfs_dict[minor_class][0],
                video_ids=class_dfs_dict[minor_class][1],
                start_ind=resample_start_ind,
                nb_extra=abs(diff))
        else:
            extra_seqs_for_minor_class = []

        print('Sampled {} extra sequences from the minor pain={} class'.format(
            len(extra_seqs_for_minor_class), minor_class
        ))

        all_seqs = no_pain_sequence_dfs + pain_sequence_dfs + extra_seqs_for_minor_class

        return all_seqs

    def prepare_image_generator_5D(self, sequence_dfs, train):
        """
        Prepare batches of frame sequences and labels,
        with help from the DataFrame with frame paths and labels.
        :param sequence_dfs: pd.DataFrame
        :param train: Boolean
        :return: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """
        nb_aug = self.aug_flip + self.aug_crop + self.aug_light
        batch_requirement = 1 + nb_aug  # Normal sequence plus augmented sequences.
        assert (self.batch_size % batch_requirement) == 0

        while True:
            if train:  # Shuffle videos between epochs.
                print('Shuffling the order of sequences.')
                random.shuffle(sequence_dfs)

            batch_index = 0
            for sequence_df in sequence_dfs:

                X_seq_list = []
                y_seq_list = []

                for seq_index, row in sequence_df.iterrows():

                    if self.config_dict['data_type'] == 'rgb':
                        x = self.get_image(row['path'])
                    if self.config_dict['data_type'] == 'of':
                        x = self.get_flow(row['of_path'])

                    y = row['pain']
                    X_seq_list.append(x)
                    y_seq_list.append(y)

                if batch_index == 0:
                    X_batch_list = []
                    y_batch_list = []

                X_batch_list.append(X_seq_list)
                y_batch_list.append(y_seq_list[0])
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
        Prepare batches of frames and labels, with help from
        the DataFrame containing frame paths and labels.
        :param df: pd.DataFrame
        :param train: Boolean
        :return: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """

        nb_frames = len(df)
        print("LEN DF (nb. of frames): ", nb_frames)
        print('Datatype: ', self.config_dict['data_type'])
        while True:
            if train:
                # Shuffle videos between epochs.
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
        flipped = []
        for img in images:
            flipped_img = self.flip_image(img)
            flipped.append(flipped_img)
        flipped_array = np.array(flipped, dtype=np.float32)
        return flipped_array

    @tf.function
    def flip_image(self, image):
        return tf.image.flip_left_right(image)

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

        y1 = offset_height / (height - 1)
        x1 = offset_width / (width - 1)
        y2 = (offset_height + target_height) / (height - 1)
        x2 = (offset_width + target_width) / (width - 1)

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
                    batch_img = np.expand_dims(img, axis=0)
                    cropped_imgs = sess.run([tf_img1], feed_dict={X: batch_img})
                    X_crops.extend(cropped_imgs)
            if self.nb_input_dims == 4:
                batch_img = np.expand_dims(images, axis=0)
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
        :param image: np.array
        :param target_height: int
        :param target_width: int
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

        y1 = offset_height / (height - 1)
        x1 = offset_width / (width - 1)
        y2 = (offset_height + target_height) / (height - 1)
        x2 = (offset_width + target_width) / (width - 1)

        boxes = np.array([[y1, x1, y2, x2]], dtype=np.float32)
        box_ind = np.array([0], dtype=np.int32)
        crop_size = np.array([width, height], dtype=np.int32)

        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, shape=(1, height, width, 3))
        tf_img1 = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batch_img = np.expand_dims(image, axis=0)
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
        clip_file = config_file['clip_list_pf'] \
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
                if filename.startswith('frame_') \
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
                if filename.startswith('flow_') \
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

    def get_y_batches_paths_from_dfs(self, sequence_dfs):

        assert(len(sequence_dfs) % self.config_dict['batch_size'] == 0)

        pain_label_list = [sdf.iloc[0]['pain'] for sdf in sequence_dfs]
        path_list = [sdf.iloc[0]['path'] for sdf in sequence_dfs]

        pain_array = np.array(pain_label_list)
        pain_array_one_hot = tf.keras.utils.to_categorical(
            pain_array, num_classes=self.config_dict['nb_labels'])
        pain_array = np.reshape(pain_array_one_hot,
                                (-1,
                                 self.config_dict['batch_size'],
                                 self.config_dict['nb_labels']))
        path_array = np.array(path_list)
        return pain_array, path_array

    def round_to_batch_size(self, data_list):
        surplus = len(data_list) % self.config_dict['batch_size']
        if surplus != 0:
            data_list = data_list[:-surplus]
        return data_list


def plot_augmentation(train, val, test, rgb, X_seq_list, flipped, cropped, shaded,
                      seq_index, batch_index, window_index):
    rows = 4
    cols = 10
    f, axarr = plt.subplots(rows, cols, figsize=(20, 10))
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


def zero_pad_list(list_to_pad, pad_length):
    list_length = len(list_to_pad)
    print('list length in zero pad: ', list_length)
    element_shape = list_to_pad[0].shape
    zeros = np.zeros(element_shape)
    nb_to_pad = pad_length - list_length

    for p in range(nb_to_pad):
        list_to_pad.append(zeros)

    return list_to_pad

