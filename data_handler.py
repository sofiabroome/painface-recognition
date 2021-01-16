import tensorflow as tf
import pandas as pd
import numpy as np
import subprocess
import helpers
import random
import time
import cv2
import re
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


DEFAULT_NPLOAD_STR = 'arr_0'
AUTOTUNE = tf.data.experimental.AUTOTUNE


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

    def batch_augment(self, x_batch, label_batch):
        clips = []
        flow_clips = []
        for b in range(self.batch_size):
            if tf.random.uniform((), minval=0, maxval=1) > 0.5:
                print('Flipping!')
                clip = [tf.image.flip_left_right(x_batch[b,0,i,:]) for i in range(self.seq_length)]
                flow_clip = [tf.image.flip_left_right(x_batch[b,1,i,:]) for i in range(self.seq_length)]
            else:
                print('Not flipping!')
                clip = [x_batch[b,0,i,:] for i in range(self.seq_length)]
                flow_clip = [x_batch[b,1,i,:] for i in range(self.seq_length)]
            clips.append(clip)
            flow_clips.append(flow_clip)
        x_batch = [clips, flow_clips]
        return x_batch, label_batch

    def augment(self, x, label):
        if tf.random.uniform((), minval=0, maxval=1) > 0.5:
            print('Flipping!')
            clip = [tf.image.flip_left_right(x[0,i,:]) for i in range(self.seq_length)]
            flow_clip = [tf.image.flip_left_right(x[1,i,:]) for i in range(self.seq_length)]
        else:
            print('Not flipping!')
            clip = [x[0,i,:] for i in range(self.seq_length)]
            flow_clip = [x[1,i,:] for i in range(self.seq_length)]
        x = [clip, flow_clip]
        return x, label

    def process_image(self, path, standardize):
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        frame = tf.io.read_file(path)
        frame = tf.image.decode_jpeg(frame, channels=self.color_channels)
        frame = normalization_layer(frame)
        if standardize:
            frame = (frame - self.pixel_mean)/self.pixel_std
        return frame

    def process_clips(self, x, label):
        clip = []
        flow_clip = []
        for i in range(self.seq_length):
            frame = self.process_image(x[0,i], standardize=True)
            flow = self.process_image(x[1,i], standardize=False)
            clip.append(frame)
            flow_clip.append(flow)
        x = [clip, flow_clip]
        return x, label

    def process_clips_paths(self, x, label, paths=None):
        clip = []
        flow_clip = []
        for i in range(self.seq_length):
            frame = self.process_image(x[0,i], standardize=True)
            flow = self.process_image(x[1,i], standardize=False)
            clip.append(frame)
            flow_clip.append(flow)
        x = [clip, flow_clip]
        if paths is not None:
            return x, label, paths
        else:
            return x, label

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
                        output_types=(tf.string, tf.uint8, tf.string))
                    dataset = dataset.map(self.process_clips_paths, num_parallel_calls=AUTOTUNE)
                    dataset = dataset.prefetch(AUTOTUNE)
                    dataset = dataset.batch(self.batch_size)
                else:
                    dataset = tf.data.Dataset.from_generator(
                        lambda: self.prepare_2stream_image_generator_5D(sequence_dfs, train),
                        output_types=(tf.string, tf.uint8))

                    dataset = dataset.map(self.process_clips, num_parallel_calls=AUTOTUNE)
                    
                    if train and (self.aug_flip == 1):
                        dataset = dataset.map(self.augment, num_parallel_calls=AUTOTUNE)
                    dataset = dataset.prefetch(AUTOTUNE)
                    dataset = dataset.batch(self.batch_size)

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

    def features_to_dataset(self, subjects, split):
        bs = self.config_dict['video_batch_size_train'] if split == 'train' \
            else self.config_dict['video_batch_size_test']
        subj_codes = []
        for subj in subjects:
            code = self.all_subjects_df[
                self.all_subjects_df['subject'] == subj]['code']
            subj_codes.append(code.values[0])

        if self.config_dict['tfrecords']:
            base_path = self.config_dict['data_path'] + self.config_dict['tfr_file']
            file_paths = [(base_path + '_{}.tfrecords'.format(sc)) for sc in subj_codes]
            dataset = tf.data.Dataset.from_tensor_slices(file_paths)
            dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
                cycle_length=8, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=False)
            dataset = dataset.map(self.parse_fn, num_parallel_calls=AUTOTUNE)

        else:
            dataset = tf.data.Dataset.from_generator(
                lambda: self.generate_features(subject_codes=subj_codes,
                                               split=split),
                output_types=(tf.float32, tf.float32, tf.int32, tf.string),
                output_shapes=(tf.TensorShape([None, None]),
                               tf.TensorShape([None, 2]),
                               tf.TensorShape([None, 2]),
                               tf.TensorShape([]))
            )
        print('Shuffling dataset...')
        dataset = dataset.shuffle(
            self.config_dict['shuffle_buffer'], reshuffle_each_iteration=True)
        print('Split: {}, batch size: {}'.format(split, bs))
        dataset = dataset.batch(bs, drop_remainder=True)
        dataset = dataset.prefetch(AUTOTUNE)
        print(dataset)
        return dataset

    def parse_fn(self, proto):
    
        # Define the tfrecord again. The sequence was saved as a string.
        keys_to_features = {
            'nb_clips': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'features': tf.io.FixedLenFeature([], tf.string),
            'preds': tf.io.FixedLenFeature([], tf.string),
            'labels': tf.io.FixedLenFeature([], tf.string),
            'video_id': tf.io.FixedLenFeature([], tf.string),
        }
    
        # Load one example
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    
        video_ID = parsed_features['video_id']
        feats = tf.io.parse_tensor(parsed_features['features'], out_type=tf.float32)
        preds = tf.io.parse_tensor(parsed_features['preds'], out_type=tf.float32)
        labels = tf.io.parse_tensor(parsed_features['labels'], out_type=tf.int32)
    
        # feats = tf.ensure_shape(feats, [self.config_dict['video_pad_length'], self.config_dict['feature_dim']])
        # preds = tf.ensure_shape(preds, [self.config_dict['video_pad_length'], self.config_dict['nb_labels']])
        # labels = tf.ensure_shape(labels, [self.config_dict['video_pad_length'], self.config_dict['nb_labels']])
    
        feats = tf.ensure_shape(feats, [self.config_dict['video_pad_length'],
                                        self.config_dict['feature_dim']])
        preds = tf.ensure_shape(preds, [self.config_dict['video_pad_length'],
                                        self.config_dict['nb_labels']])
        labels = tf.ensure_shape(labels, [self.config_dict['video_pad_length'],
                                          self.config_dict['nb_labels']])
    
        # feats = tf.cast(feats, tf.float32)
        # preds = tf.cast(preds, tf.float32)
        # labels = tf.cast(labels, tf.int32)
    
        return feats, preds, labels, video_ID

    def generate_features(self,
                          subject_codes,
                          split):
        """
        Load features from file (per video).
        :param subject_codes: [str]
        :param features_folder: str
        :yield: batch features, batch preds, batch labels
        """
        if split == 'train':
            feature_folder = self.config_dict['train_video_features_folder']
        if split == 'val':
            feature_folder = self.config_dict['val_video_features_folder']
        if split == 'test':
            feature_folder = self.config_dict['test_video_features_folder']

        path_to_features = self.config_dict['data_path'] + feature_folder
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
            # print('\n video_id:', video_id)
            # print('shapes: ', f_shape, preds.shape, labels.shape)
            assert preds.shape[0] == f_shape[0]
            assert labels.shape[0] == f_shape[0]
            yield feats, preds, labels, video_id

    def prepare_video_features(self, features, zero_pad=False):
        save_folder = self.config_dict['data_path'] + self.config_dict['save_video_features_folder']
        pad_length = self.config_dict['video_pad_length']
        if not os.path.exists(save_folder):
            subprocess.call(['mkdir', save_folder])

        nb_clip_batches = features[DEFAULT_NPLOAD_STR].shape[0]
        dict_of_dicts = {}  # key will be video ID, value will be to_save_dict
        batches = features[DEFAULT_NPLOAD_STR]
        batches = batches.tolist()  # For faster iteration

        for clip_batch in range(nb_clip_batches):
            print('Batch {}/{}'.format(clip_batch, nb_clip_batches))
            # Get the data from one batch (8 short clips).
            st = time.time()
            clip_batch_feats = batches[clip_batch]['features'].numpy()
            clip_batch_preds = batches[clip_batch]['preds'].numpy()
            clip_batch_labels =batches[clip_batch]['y'].numpy()
            clip_batch_paths = batches[clip_batch]['paths'].numpy()
            print('Time taken %.6f' % (time.time() - st))

            # Iterate over the short clips in the batch.
            for ind, path in enumerate(clip_batch_paths):
                video_id = get_video_id_from_frame_path(str(path))
                if clip_batch == 0 and ind == 0:
                    print('start, video id: ', video_id)
                    old_video_id = video_id
                    same_video_features = []
                    same_video_preds = []
                    same_video_labels = []
                    same_video_paths = []

                if video_id != old_video_id:
                    if old_video_id in dict_of_dicts:
                        print('Already had one for: ', old_video_id)
                        # print('Saving with resampling.\n')
                        print('Saving without resampling.\n')
                        continue
                    feats, preds, labels, paths = prepare_fplp(same_video_features,
                                                               same_video_preds,
                                                               same_video_labels,
                                                               same_video_paths,
                                                               pad_length=pad_length,
                                                               zero_pad=zero_pad)

                    length = feats.shape[0]
                    subject = old_video_id[0]
                    print('Saving features from old_video_id:', old_video_id)
                    print('shapes: ', feats.shape, preds.shape, labels.shape, paths.shape)
                    to_save_dict = put_in_dict(feats, preds, labels, paths, length, subject)
                    if old_video_id in dict_of_dicts:
                        print('Already had one for: ', old_video_id)
                        # print('Saving with resampling.\n')
                        print('Saving without resampling.\n')
                        continue
                        # print(labels, '\n')
                        dict_to_merge_with = dict_of_dicts[old_video_id]
                        merged_dict, length = mergesort_features_into_dict(
                            old_video_id, dict_to_merge_with, to_save_dict, pad_length, zero_pad)
                        dict_of_dicts[old_video_id] = merged_dict
                    else:
                        dict_of_dicts[old_video_id] = to_save_dict

                    print('\n New video id: ', video_id)
                    same_video_features = []
                    same_video_preds = []
                    same_video_labels = []
                    same_video_paths = []
                    old_video_id = video_id

                same_video_features.append(clip_batch_feats[ind])
                same_video_preds.append(clip_batch_preds[ind])
                same_video_labels.append(clip_batch_labels[ind])
                same_video_paths.append(path)

        # Finally also use the last one (video ID doesn't change)
        feats, preds, labels, paths = prepare_fplp(same_video_features,
                                                   same_video_preds,
                                                   same_video_labels,
                                                   same_video_paths,
                                                   pad_length=pad_length,
                                                   zero_pad=zero_pad)
        length = feats.shape[0]
        subject = video_id[0]
        print('\n Saving last video_id:', video_id)
        print('shapes: ', feats.shape, preds.shape, labels.shape, paths.shape)

        to_save_dict = put_in_dict(feats, preds, labels, paths, length, subject)

        if video_id in dict_of_dicts:
            print('Already had one for: ', video_id)
            print('Saving without resampling.\n')
            # print('Saving with resampling.\n')
            # dict_to_merge_with = dict_of_dicts[video_id]
            # merged_dict, length = mergesort_features_into_dict(
            #     video_id, dict_to_merge_with, to_save_dict, pad_length, zero_pad)
            # dict_of_dicts[video_id] = merged_dict
        else:
            dict_of_dicts[video_id] = to_save_dict

        big_list = []
        for video_id, to_save_dict in dict_of_dicts.items():
            subject = to_save_dict['subject']
            length = to_save_dict['length']
            video_list = [subject, video_id, length]
            save_filename = save_folder + video_id + '.npz'
            to_save_dict = np.array(to_save_dict)
            np.savez_compressed(save_filename, to_save_dict)
            big_list.append(video_list)

        col_headers = ['subject', 'video_id', 'length']
        video_df = pd.DataFrame(big_list, columns=col_headers)
        video_df.to_csv(save_folder + 'summary.csv')

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

            for sequence_df in sequence_dfs:

                X_seq_list = []
                y_seq_list = []
                flow_seq_list = []

                for seq_index, row in sequence_df.iterrows():

                    if (seq_index % self.config_dict['rgb_period']) == 0:
                        x = row['path']
                        y = row['pain']
                        X_seq_list.append(x)
                        y_seq_list.append(y)

                    if (seq_index % self.config_dict['flow_period']) == 0:
                        flow = row['of_path']
                        if self.config_dict['rgb_period'] > 1:
                            # We only want the first two channels of the flow
                            flow = np.take(flow, [0, 1], axis=2)  # Simonyan type input
                        flow_seq_list.append(flow)

                # *We only have per-clip labels, so the pain levels should not differ.
                assert (len(set(y_seq_list)) == 1)
                if self.config_dict['rgb_period'] > 1:
                    #TODO deprecated. move this to tf.dataset.map preprocessing instead.
                    flow_seq_list = np.array(flow_seq_list)
                    flow_seq_list = np.reshape(np.array(flow_seq_list),
                                               (-1, self.image_size[0], self.image_size[1]))
                    X_seq_list = np.reshape(np.array(X_seq_list),
                                            (self.image_size[0], self.image_size[1], -1))

                X_array = np.array(X_seq_list, dtype=np.dtype('U'))
                flow_array = np.array(flow_seq_list, dtype=np.dtype('U'))
                y_array = np.array(y_seq_list[0], dtype=np.uint8)
                if self.nb_labels == 2:
                    y_array = tf.keras.utils.to_categorical(y_array, num_classes=self.nb_labels)
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
            for sequence_df in sequence_dfs:

                X_seq_list = []
                y_seq_list = []
                flow_seq_list = []
                path_seq_list = []

                for seq_index, row in sequence_df.iterrows():

                    if (seq_index % self.config_dict['rgb_period']) == 0:
                        # x = self.get_image(row['path'])
                        x = row['path']
                        y = row['pain']
                        X_seq_list.append(x)
                        y_seq_list.append(y)
                        path_seq_list.append(row['path'])  # Only save one (last) path per seq

                    if (seq_index % self.config_dict['flow_period']) == 0:
                        # flow = self.get_flow(row['of_path'])
                        flow = row['of_path']
                        if self.config_dict['rgb_period'] > 1:
                            # We only want the first two channels of the flow
                            flow = np.take(flow, [0, 1], axis=2)  # Simonyan type input
                        flow_seq_list.append(flow)

                # *We only have per-clip labels, so the pain levels should not differ.
                assert (len(set(y_seq_list)) == 1)

                X_array = np.array(X_seq_list, dtype=np.dtype('U'))
                y_array = np.array(y_seq_list[0], dtype=np.uint8)
                flow_array = np.array(flow_seq_list, dtype=np.dtype('U'))
                path_array = np.array(path_seq_list)
                if self.nb_labels == 2:
                    y_array = tf.keras.utils.to_categorical(y_array, num_classes=self.nb_labels)
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
                number_of_windows = (last_valid_end_index+1) // window_stride
                # print('Number of windows', number_of_windows)
                start_indices = [(start_ind + window_index * window_stride)
                                 for window_index in range(number_of_windows)]
            else:  # Resampling for minor class
                print('\nComputing start indices for resampling from the following df...\n')
                print('video_frame_df.head():')
                print(video_frame_df.head(), '\n')
                print('Frames in video: {}, nb per video: {}, last valid start {}'.format(
                    nb_frames_in_video, nb_per_video, last_valid_start_index))
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
    print('nb_to_pad: ', nb_to_pad)
    
    for p in range(nb_to_pad):
        list_to_pad.append(zeros)

    return list_to_pad


def put_in_dict(feats, preds, labels, paths, length, subject):
    to_save_dict = {'features': feats, 'preds': preds, 'labels': labels,
                    'paths': paths, 'length': length, 'subject': subject}
    return to_save_dict


def prepare_fplp(same_video_features, same_video_preds, same_video_labels, same_video_paths,
                 pad_length, zero_pad=False):
    """
    :param same_video_features: [np.array]
    :param same_video_preds: [np.array]
    :param same_video_labels: [np.array]
    :param same_video_paths: [str]
    :param pad_length: int
    :param zero_pad: bool
    :return: (np.array, np.array, np.array, np.array)
    """
    if zero_pad:
        same_video_features = zero_pad_list(same_video_features, pad_length)
        same_video_preds = zero_pad_list(same_video_preds, pad_length)
        same_video_labels = zero_pad_list(same_video_labels, pad_length)
    print('Reshaping and arraying...')
    st = time.time()
    feats = np.array(same_video_features)
    f_shape = feats.shape
    if len(f_shape) == 3:
        feats = np.reshape(feats, [f_shape[0], f_shape[1] * f_shape[2]])
    preds = np.array(same_video_preds)
    labels = np.array(same_video_labels)
    paths = np.array(same_video_paths)
    assert preds.shape[0] == feats.shape[0]
    assert labels.shape[0] == feats.shape[0]
    print('Time taken for this prep %.4f' % (time.time() - st))
    # assert paths.shape[0] == feats.shape[0]
    return feats, preds, labels, paths


def mergesort_features_into_dict(video_id, d1, d2, pad_length, zero_pad):
    """
    Both dicts have keys: features, preds, labels, paths, length, subject.
    Need to mergesort them using the paths.
    :param video_id: str
    :param d1: dict
    :param d2: dict
    :param pad_length: int
    :return: dict
    """
    # All paths are named frame_xxxxxx.jpg, hence take -11:-5
    d1_internal_inds_frame_inds = [(1, ind, int(str(frame_path)[-11:-5])) for ind, frame_path in enumerate(d1['paths'])]
    d2_internal_inds_frame_inds = [(2, ind, int(str(frame_path)[-11:-5])) for ind, frame_path in enumerate(d2['paths'])]

    # Example result:
    # d1_internal_inds_frame_inds = [(1, 0, 1), (1, 1, 11), (1, 2, 21), (1, 3, 31), (1, 4, 41)]
    # d2_internal_inds_frame_inds = [(2, 0, 6), (2, 1, 36), (2, 2, 56), (2, 3, 86), (2, 4, 106)]

    # So the internal inds should be updated according to the frame inds.
    # (Need to keep track of which is which, hence have identifiers in each tuple before sorting.)

    # Concatenate and sort on the frame index
    combined = d1_internal_inds_frame_inds + d2_internal_inds_frame_inds
    combined.sort(key=lambda x: x[2])

    same_video_feats = []
    same_video_preds = []
    same_video_labels = []
    same_video_paths = []

    for id, ind, frame_ind in combined:
        data = d1 if id == 1 else d2
        same_video_feats.append(data['features'][ind])
        same_video_preds.append(data['preds'][ind])
        same_video_labels.append(data['labels'][ind])
        same_video_paths.append(data['paths'][ind])

    feats, preds, labels, paths = prepare_fplp(same_video_features=same_video_feats,
                                               same_video_preds=same_video_preds,
                                               same_video_labels=same_video_labels,
                                               same_video_paths=same_video_paths,
                                               pad_length=pad_length,
                                               zero_pad=zero_pad)

    length = len(same_video_feats)
    subject = video_id[0]
    merged_dict = put_in_dict(feats, preds, labels, paths, length, subject)

    return merged_dict, length



