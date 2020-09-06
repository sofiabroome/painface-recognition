import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import cv2
import os

from keras.utils import np_utils

from helpers import process_image, split_string_at_last_occurence_of_certain_char


class DataHandler:
    def __init__(self, path, of_path, data_columns, config_dict, color):
        """
        Constructor for the DataHandler.
        :param path: str
        :param config_dict: dict
        :param color: bool
        :param nb_labels: int
        """
        self.path = path
        self.of_path = of_path
        self.data_columns = data_columns
        self.image_size = config_dict['input_width'], config_dict['input_height']
        self.seq_length = config_dict['seq_length']
        self.seq_stride = config_dict['seq_stride']
        self.batch_size = config_dict['batch_size']
        self.color = color
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

    def get_generator(self, df, train):
        """
        Get a generator for a DataFrame, appropriate for the model.
        :param df: pd.DataFrame
        :param train:  boolean
        :return: generator
        """

        if self.config_dict['nb_input_dims'] == 5:
            if '2stream' in self.config_dict['model']:
                generator = self.prepare_2stream_image_generator_5D(
                    df, train
                )
            else:
                generator = self.prepare_image_generator_5D(
                    df, train
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

        return generator

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
                flow = self.get_image(row['of_path'])
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
                        y_array = np_utils.to_categorical(y_array, num_classes=self.nb_labels)
                    y_array = np.reshape(y_array, (self.batch_size, self.nb_labels))
                    batch_index = 0
                    # print(X_array.shape, flow_array.shape, y_array.shape)
                    yield [X_array, flow_array], [y_array]

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
                        print('First frame. Set oldname=vidname')
                        old_vid_seq_name = vid_seq_name  # This variable is set once
                        this_index += 1
                    
                    if vid_seq_name != old_vid_seq_name:
                        seq_index = 0
                        old_vid_seq_name = vid_seq_name
                        break  # Skip this one and jump to next window

                    if (seq_index % self.config_dict['rgb_period']) == 0:
                        x = self.get_image(row['path'])
                        X_seq_list.append(x)
                        y = row['pain']
                        y_seq_list.append(y)

                    if (seq_index % self.config_dict['flow_period']) == 0:
                        flow = self.get_image(row['of_path'])
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
                    if self.config_dict['rgb_period'] > 1:
                        flow_seq_list = np.array(flow_seq_list)
                        flow_seq_list = np.reshape(np.array(flow_seq_list),
                                                  (-1, self.image_size[0], self.image_size[1]))
                        X_seq_list = np.reshape(np.array(X_seq_list),
                                               (self.image_size[0], self.image_size[1], -1))
                        
                    X_batch_list.append(X_seq_list)
                    y_batch_list.append(y_seq_list)
                    flow_batch_list.append(flow_seq_list)
                    batch_index += 1
                    seq_index = 0
                    
                    if train and (self.aug_flip == 1):
                        # Flip both RGB and flow arrays
                        X_seq_list_flipped = self.flip_images(X_seq_list)
                        flow_seq_list_flipped = self.flip_images(flow_seq_list)
                        # Append to the respective batch lists
                        X_batch_list.append(X_seq_list_flipped)
                        y_batch_list.append(y_seq_list)
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
                        y_batch_list.append(y_seq_list)
                        flow_batch_list.append(flow_seq_list_cropped)
                        batch_index += 1

                    if train and (self.aug_light == 1):
                        # Flip both RGB and flow arrays
                        X_seq_list_shaded = self.add_gaussian_noise(X_seq_list)
                        flow_seq_list_shaded = self.add_gaussian_noise(flow_seq_list)
                        # Append to the respective batch lists
                        X_batch_list.append(X_seq_list_shaded)
                        y_batch_list.append(y_seq_list)
                        flow_batch_list.append(flow_seq_list_shaded)
                        batch_index += 1

                    # if train:
                    #     plot_augmentation(train, val, test, evaluate, 1, X_seq_list, X_seq_list_flipped, X_seq_list_cropped,
                    #                       X_seq_list_shaded, seq_index, batch_index, window_index)
                    #     plot_augmentation(train, val, test, evaluate, 0, flow_seq_list, flow_seq_list_flipped, flow_seq_list_cropped,
                    #                       flow_seq_list_shaded, seq_index, batch_index, window_index)

                if batch_index % self.batch_size == 0 and not batch_index == 0:
                    X_array = np.array(X_batch_list, dtype=np.float32)
                    y_array = np.array(y_batch_list, dtype=np.uint8)
                    flow_array = np.array(flow_batch_list, dtype=np.float32)
                    if self.nb_labels == 2:
                        y_array = np_utils.to_categorical(y_array, num_classes=self.nb_labels)
                        y_array = np.reshape(y_array, (self.batch_size, -1, self.nb_labels))
                    else:
                        y_array = np.reshape(y_array, (self.batch_size, -1, self.nb_labels))
                    if self.config_dict['rgb_period'] > 1:
                        y_array = np.reshape(y_array, (self.batch_size, self.nb_labels))
                    batch_index = 0
                    # print(X_array.shape, flow_array.shape, y_array.shape)
                    yield [X_array, flow_array], [y_array]

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

        ws = self.seq_length  # "Window size" in a sliding window.
        ss = self.seq_stride  # Stride for the extracted windows
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
            # Shuffle blocks between epochs if during training.
            if train:
                df = shuffle_blocks(df, 'video_id')
            batch_index = 0
            for window_index in range(nw):
                start = window_index * ss
                stop = start + ws
                rows = df.iloc[start:stop]  # A new dataframe for the window in question.

                X_seq_list = []
                y_seq_list = []

                for index, row in rows.iterrows():
                    vid_seq_name = row['video_id']

                    if this_index == 0:
                        print('First frame. Set oldname=vidname.')
                        old_vid_seq_name = vid_seq_name # Set this variable (only once).
                        this_index += 1

                    if vid_seq_name != old_vid_seq_name:
                        seq_index = 0
                        old_vid_seq_name = vid_seq_name
                        break  # In that case want to jump to the next window.

                    if self.config_dict['data_type'] == 'rgb':
                        x = self.get_image(row['path'])
                    if self.config_dict['data_type'] == 'of':
                        x = self.get_image(row['of_path'])
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
                    y_batch_list.append(y_seq_list)
                    seq_index = 0
                    batch_index += 1

                    if train and (self.aug_flip == 1):
                        X_seq_list_flipped = self.flip_images(X_seq_list)
                        X_batch_list.append(X_seq_list_flipped)
                        y_batch_list.append(y_seq_list)
                        batch_index += 1

                    if train and (self.aug_crop == 1):
                        crop_size = 99
                        X_seq_list_cropped = self.random_crop_resize(X_seq_list,
                                                                     crop_size, crop_size)
                        X_batch_list.append(X_seq_list_cropped)
                        y_batch_list.append(y_seq_list)
                        batch_index += 1

                    if train and (self.aug_light == 1):
                        X_seq_list_shaded = self.add_gaussian_noise(X_seq_list)
                        X_batch_list.append(X_seq_list_shaded)
                        y_batch_list.append(y_seq_list)
                        batch_index += 1

                    # if train:
                    #     plot_augmentation(train, val, test, evaluate, 1, X_seq_list, X_seq_list_flipped, X_seq_list_cropped,
                    #                       X_seq_list_shaded, seq_index, batch_index, window_index)

                if batch_index % self.batch_size == 0 and not batch_index == 0:
                    X_array = np.array(X_batch_list, dtype=np.float32)
                    y_array = np.array(y_batch_list, dtype=np.uint8)
                    if self.nb_labels == 2:
                        y_array = np_utils.to_categorical(y_array, num_classes=self.nb_labels)
                        y_array = np.reshape(y_array, (self.batch_size, -1, self.nb_labels))
                    batch_index = 0
                    yield (X_array, y_array)

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
                    x = self.get_image(row['of_path'])
                y = row['pain']
                X_list.append(x)
                y_list.append(y)
                batch_index += 1

                if batch_index % self.batch_size == 0:
                    X_array = np.array(X_list, dtype=np.float32)
                    y_array = np.array(y_list, dtype=np.uint8)
                    y_array = np_utils.to_categorical(y_array,
                                                      num_classes=self.nb_labels)
                    batch_index = 0
                    yield (X_array, y_array)

    def get_image(self, path):
        channels = 3 if self.color else 1
        im = process_image(path, (self.image_size[0], self.image_size[1], channels))
        return im

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

        ch = 3 if self.color else 1
        row, col = self.image_size
    
        mean = 0
        sigma = 0.5

        imw_a = 0.55
        imw_b = 0.7
        im_weight = (imw_b - imw_a) * np.random.random() + imw_a

        now_a = 0.2
        now_b = 0.4
        noise_weight = (now_b - now_a) * np.random.random() + now_a

        gaussian = np.random.normal(mean, sigma, (col, row, ch)).astype(np.float32)

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
        ch = 3 if self.color else 1
        row, col = self.image_size
    
        mean = 0
        sigma = 0.5

        imw_a = 0.55
        imw_b = 0.7
        im_weight = (imw_b - imw_a) * np.random.random() + imw_a

        now_a = 0.2
        now_b = 0.4
        noise_weight = (now_b - now_a) * np.random.random() + now_a

        gaussian = np.random.normal(mean, sigma, (col, row, ch)).astype(np.float32)

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

    # TODO Merge the two below functions (subject_to_df and save_OF_paths_to_df, same functionality)

    def subject_to_df(self, subject_id, dataset, config_file):
        """
        Create a DataFrame with all the frames with annotations from a csv-file.
        :param subject_id: int
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
                total_path = os.path.join(path, filename)
                print(total_path)
                vid_id = get_video_id_stem_from_path(path, dataset)
                csv_row = df_csv.loc[df_csv['video_id'] == vid_id]
                if csv_row.empty:
                    continue
                if '.jpg' in filename or '.png' in filename:
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
        :return: pd.DataFrame
        """
        c = 0  # Per subject frame counter.
        per_clip_frame_counter = 0
        old_path = 'NoPath'
        root_of_path = os.path.join(
            self.dataset_of_path_dict[dataset], subject_id)
        of_path_list = []

        # Walk through all the files in the of-folders and put them in a
        # list, in order (the same order they were extracted in.)

        for path, dirs, files in sorted(os.walk(root_of_path)):
            print(path)
            video_id = get_video_id_from_path(path)
            nb_frames_in_clip = len(subject_df.loc[subject_df['path'].str.contains(video_id)])
            print(video_id)
            if old_path != path and c != 0:  # If entering a new folder
                per_clip_frame_counter = 0
                if '1fps' in self.of_path or 'ShoulderPain' in self.of_path: # To match the #pictures with #of I disregard the first frame.
                    subject_df.drop(c, inplace=True)  # Delete first element
                    subject_df.reset_index(drop=True, inplace=True)  # And adjust the index
            old_path = path
            for filename in sorted(files):
                total_path = os.path.join(path, filename)
                if '.npy' in filename or '.jpg' in filename:        # (If it's an optical flow-array.)
                    if per_clip_frame_counter > nb_frames_in_clip:  # This can probably be removed but will
                        break                                       # leave it here for now.
                    of_path_list.append(total_path)
                    c += 1
                    per_clip_frame_counter += 1

        # Now extend subject_df to contain both rgb and OF paths,
        # and then return whole thing.
        nb_of_paths = len(of_path_list)
        nb_rgb_frames = len(subject_df)
        if nb_rgb_frames != nb_of_paths:
            diff = nb_rgb_frames - nb_of_paths
            print("Differed by:", diff)
            # (They should only differ by one row.
            # Else an error should be raised when concatenating.)
            if diff < 0: # If the of-df was larger, reduce it
                of_path_list = of_path_list[:diff]
            else:  # Vice versa with subject-df
                subject_df = subject_df[:-diff]
        try:
            subject_df.loc[:, 'of_path'] = pd.Series(of_path_list)
            subject_df.loc['train'] = -1
        except AssertionError:
            print('Horse df and OF_df were not the same length and could not'
                  'be concatenated. Even despite having removed the last'
                  'element of subject df which should be 1 longer.')

        return subject_df

    def _get_images_from_df(self, df):
        """
        Get the images as arrays from all the paths in a DataFrame.
        :param df: pd.DataFrame
        :return: [np.ndarray]
        """
        images = []
        if self.color:
            channels = 3
        else:
            channels = 1
        for path in df['path']:
            im = process_image(path,
                               (self.image_size[0],
                                self.image_size[1],
                                channels))
            images.append(im)
        return images

def plot_augmentation(train, val, test, evaluate, rgb, X_seq_list, flipped, cropped, shaded,
                      seq_index, batch_index, window_index):
    rows = 4
    cols = 10
    f, axarr = plt.subplots(rows, cols, figsize=(20,10))
    for i in range(0, rows):
        for j in range(0, cols):
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
            # axarr[i, j].set_aspect('equal')
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
    # plt.axis('off')
    # plt.tight_layout()
    if train:
        partition = 1
    elif val:
        partition = 2
    elif test:
        partition = 3
    else:
        partition = 4
    plt.savefig('seq_{}_batch_{}_wi_{}_part_{}_rgb_{}.png'.format(seq_index,
                                                           batch_index,
                                                           window_index,
                                                           partition,
                                                           rgb))
    plt.close()


def get_video_id_stem_from_path(path, dataset):
    _, vid_id = split_string_at_last_occurence_of_certain_char(path, '/')
    if dataset == 'pf':
        nb_underscore = vid_id.count('_')
        if nb_underscore > 1:
            vid_id, _ = split_string_at_last_occurence_of_certain_char(vid_id, '_')
    return vid_id


def get_video_id_from_path(path):
    _, vid_id = split_string_at_last_occurence_of_certain_char(path, '/')
    return vid_id

def get_video_id_from_frame_path(path):
    path_left, frame_id = split_string_at_last_occurence_of_certain_char(path, '/')
    _, vid_id = split_string_at_last_occurence_of_certain_char(path_left, '/')
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

