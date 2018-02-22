import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import cv2
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from os.path import join

from helpers import split_string_at_last_occurence_of_certain_char
from image_processor import process_image

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
eval_datagen = ImageDataGenerator()


class DataHandler:
    def __init__(self, path, of_path, image_size, seq_length,
                 seq_stride, batch_size, color, nb_labels,
                 aug_flip, aug_crop, aug_light):
        """
        Constructor for the DataHandler.
        :param path: str
        :param image_size: (int, int)
        :param seq_length: int
        :param color: bool
        :param nb_labels: int
        """
        self.path = path
        self.of_path = of_path
        self.image_size = image_size
        self.seq_length = seq_length
        self.seq_stride = seq_stride
        self.batch_size = batch_size
        self.color = color
        self.nb_labels = nb_labels
        self.aug_flip = aug_flip
        self.aug_crop = aug_crop
        self.aug_light = aug_light

    def prepare_generator_2stream(self, df, train, val, test, evaluate):
        """
        Prepare the frames into labeled train and test sets, with help from the
        DataFrame with .jpg-paths and labels for train and pain.
        :param df: pd.DataFrame
        :param train: Boolean
        :param val: Boolean
        :param test: Boolean
        :param evaluate: Boolean
        :return: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """

        print("LEN DF:")
        print(len(df))
        while True:
            if train:
                # Shuffle blocks between epochs.
                df = shuffle_blocks(df, 'Video_ID')
            batch_index = 0
            for index, row in df.iterrows():
                if batch_index == 0:
                    X_batch_list = []
                    y_batch_list = []
                    flow_batch_list = []
                x = self.get_image(row['Path'])
                x /= 255
                y = row['Pain']
                flow = np.load(row['OF_Path'])
                extra_channel = np.zeros((flow.shape[0], flow.shape[1], 1))
                flow = np.concatenate((flow, extra_channel), axis=2)
                X_batch_list.append(x)
                y_batch_list.append(y)
                flow_batch_list.append(flow)
                batch_index += 1

                if batch_index % self.batch_size == 0:
                    X_array = np.array(X_batch_list, dtype=np.float32)
                    y_array = np.array(y_batch_list, dtype=np.uint8)
                    flow_array = np.array(flow_batch_list, dtype=np.float32)
                    # if self.nb_labels != 2:
                    y_array = np_utils.to_categorical(y_array, num_classes=self.nb_labels)
                    y_array = np.reshape(y_array, (self.batch_size, self.nb_labels))
                    batch_index = 0
                    yield [X_array, flow_array], [y_array]

    def prepare_2stream_image_generator_5D(self, df, train, val, test, evaluate):
        """
        Prepare the frames into labeled train and test sets, with help from the
        DataFrame with .jpg-paths and labels for train and pain.
        :param df: pd.DataFrame
        :param train: Boolean
        :param val: Boolean
        :param test: Boolean
        :param evaluate: Boolean
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
                df = shuffle_blocks(df, 'Video_ID')
            batch_index = 0
            for window_index in range(nw):
                start = window_index * ss
                stop = start + ws
                rows = df.iloc[start:stop]  # A new dataframe for the window in question.

                X_seq_list = []
                y_seq_list = []
                flow_seq_list = []

                for index, row in rows.iterrows():
                    vid_seq_name = row['Video_ID']

                    if this_index == 0:
                        print('First frame. Set oldname=vidname')
                        old_vid_seq_name = vid_seq_name  # This variable is set once
                        this_index += 1
                    
                    if vid_seq_name != old_vid_seq_name:
                        seq_index = 0
                        old_vid_seq_name = vid_seq_name
                        break  # Skip this one and jump to next window

                    x = self.get_image(row['Path'])
                    y = row['Pain']
                    flow = self.get_image(row['OF_Path'])

                    # Concatenate a third channel in order to comply w RGB images
                    # NOTE: If OF-path has 'magnitude' in it, no concatenation is needed and it already has 3 channels.
                    # Either just zeros, or the magnitude (can load magnitude directly now from file)
                    # extra_channel = np.zeros((flow.shape[0], flow.shape[1], 1))
                    # flow = np.concatenate((flow, extra_channel), axis=2)

                    X_seq_list.append(x)
                    y_seq_list.append(y)
                    flow_seq_list.append(flow)
                    seq_index += 1

                if batch_index == 0:
                    X_batch_list = []
                    y_batch_list = []
                    flow_batch_list = []

                if seq_index == self.seq_length:
                    X_batch_list.append(X_seq_list)
                    y_batch_list.append(y_seq_list)
                    flow_batch_list.append(flow_seq_list)
                    batch_index += 1
                    seq_index = 0
                    
                    if self.aug_flip:
                        # Flip both RGB and flow arrays
                        X_seq_list_flipped = self.flip_images(X_seq_list)
                        flow_seq_list_flipped = self.flip_images(flow_seq_list)
                        # Append to the respective batch lists
                        X_batch_list.append(X_seq_list_flipped)
                        y_batch_list.append(y_seq_list)
                        flow_batch_list.append(flow_seq_list_flipped)
                        batch_index += 1

                if batch_index % self.batch_size == 0 and not batch_index == 0:
                    X_array = np.array(X_batch_list, dtype=np.float32)
                    y_array = np.array(y_batch_list, dtype=np.uint8)
                    flow_array = np.array(flow_batch_list, dtype=np.float32)
                    if self.nb_labels == 2:
                        y_array = np_utils.to_categorical(y_array, num_classes=self.nb_labels)
                        y_array = np.reshape(y_array, (self.batch_size, -1, self.nb_labels))
                    else:
                        y_array = np.reshape(y_array, (self.batch_size, -1, self.nb_labels))
                    batch_index = 0
                    yield [X_array, flow_array], [y_array]

    def prepare_image_generator_5D(self, df, data_type, train, val, test, eval):
        """
        Prepare the frames into labeled train and test sets, with help from the
        DataFrame with .jpg-paths and labels for train and pain.
        :param df: pd.DataFrame
        :param data_type: str ['rgb' || 'of']
        :param train: Boolean
        :param val: Boolean
        :param test: Boolean
        :param eval: Boolean
        :return: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """
        nb_frames = len(df)
        print("LEN DF, in prep_5d(): ", nb_frames)

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
            # Shuffle blocks between epochs if during training.
            if train:
                df = shuffle_blocks(df, 'Video_ID')
            batch_index = 0
            for window_index in range(nw):
                start = window_index * ss
                stop = start + ws
                rows = df.iloc[start:stop]  # A new dataframe for the window in question.

                X_seq_list = []
                y_seq_list = []

                for index, row in rows.iterrows():
                    vid_seq_name = row['Video_ID']

                    if this_index == 0:
                        print('First frame. Set oldname=vidname.')
                        old_vid_seq_name = vid_seq_name # Set this variable (only once).
                        this_index += 1

                    if vid_seq_name != old_vid_seq_name:
                        seq_index = 0
                        # print('New sequence. Settin seq ind to 0 and start on new.')
                        old_vid_seq_name = vid_seq_name
                        break  # In that case want to jump to the next window.

                    if data_type == 'rgb':
                        x = self.get_image(row['Path'])
                    if data_type == 'of':
                        x = np.load(row['OF_Path'])
                        # If no magnitude:
                        # extra_channel = np.zeros((x.shape[0], x.shape[1], 1))
                        # x = np.concatenate((x, extra_channel), axis=2)
                    y = row['Pain']
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

                    if self.aug_flip:
                        X_seq_list_flipped = self.flip_images(X_seq_list)
                        X_batch_list.append(X_seq_list_flipped)
                        y_batch_list.append(y_seq_list)
                        batch_index += 1

                    if self.aug_crop:
                        crop_size = 99
                        X_seq_list_cropped = self.random_crop_resize(X_seq_list,
                                                                     crop_size, crop_size)
                        X_batch_list.append(X_seq_list_cropped)
                        y_batch_list.append(y_seq_list)
                        batch_index += 1

                    if self.aug_light:
                        X_seq_list_shaded = self.add_gaussian_noise(X_seq_list)
                        X_batch_list.append(X_seq_list_shaded)
                        y_batch_list.append(y_seq_list)
                        batch_index += 1

                    # plot_augmentation(X_seq_list, X_seq_list_flipped, X_seq_list_cropped,
                    #                   X_seq_list_shaded, seq_index, batch_index, window_index)

                if batch_index % self.batch_size == 0 and not batch_index == 0:
                    X_array = np.array(X_batch_list, dtype=np.float32)
                    y_array = np.array(y_batch_list, dtype=np.uint8)
                    if self.nb_labels == 2:
                        y_array = np_utils.to_categorical(y_array, num_classes=self.nb_labels)
                        y_array = np.reshape(y_array, (self.batch_size, -1, self.nb_labels))
                    batch_index = 0
                    yield (X_array, y_array)

    def prepare_image_generator(self, df, data_type, train, val, test, evaluate):
        """
        Prepare the frames into labeled train and test sets, with help from the
        DataFrame with .jpg-paths and labels for train and pain.
        :param df: pd.DataFrame
        :param data_type: str ['rgb' || 'of']
        :param train: Boolean
        :param val: Boolean
        :param test: Boolean
        :return: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """
        print("LEN DF:")
        print(len(df))
        print('Datatype:')
        print(data_type)
        while True:
            if train:
                # Shuffle blocks between epochs.
                df = shuffle_blocks(df, 'Video_ID')
            batch_index = 0
            for index, row in df.iterrows():
                if batch_index == 0:
                    X_list = []
                    y_list = []
                if data_type == 'rgb':
                    x = self.get_image(row['Path'])
                    # x /= 255
                if data_type == 'of':
                    x = np.load(row['OF_Path'])
                    extra_channel = np.zeros((x.shape[0], x.shape[1], 1))
                    x = np.concatenate((x, extra_channel), axis=2)
                y = row['Pain']
                X_list.append(x)
                y_list.append(y)
                batch_index += 1

                if batch_index % self.batch_size == 0:
                    # TODO Test normalization here (divide X-array by 255).
                    X_array = np.array(X_list, dtype=np.float32)
                    y_array = np.array(y_list, dtype=np.uint8)
                    y_array = np_utils.to_categorical(y_array,
                                                      num_classes=self.nb_labels)
                    if train:
                        X_array, y_array = train_datagen.flow(X_array, y_array,
                                                              batch_size=self.batch_size,
                                                              shuffle=False).next()
                    if val:
                        X_array, y_array = val_datagen.flow(X_array, y_array,
                                                            batch_size=self.batch_size,
                                                            shuffle=False).next()
                    if test:
                        X_array, y_array = test_datagen.flow(X_array, y_array,
                                                             batch_size=self.batch_size,
                                                             shuffle=False).next()
                    if evaluate:
                        X_array, y_array = eval_datagen.flow(X_array, y_array,
                                                             batch_size=self.batch_size,
                                                             shuffle=False).next()
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

        gaussian = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)

        for img in images:
            gaussian_img = cv2.addWeighted(img, im_weight, gaussian, noise_weight, 0)
            gaussian_noise_imgs.append(gaussian_img)
    
        gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.float32)
        return gaussian_noise_imgs

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
        crop_size = np.array([height, width], dtype=np.int32)
    
        X_crops = []
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, shape=(1, width, height, 3))
        tf_img1 = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for img in images:
                batch_img = np.expand_dims(img, axis = 0)
                cropped_imgs = sess.run([tf_img1], feed_dict={X: batch_img})
                X_crops.extend(cropped_imgs)
        X_crops = np.array(X_crops, dtype=np.float32)
        X_crops = np.reshape(X_crops, (self.seq_length, width, height, 3))
        return X_crops

    # TODO Merge the two below functions (horse_to_df and save_OF_paths_to_df, same functionality)

    def horse_to_df(self, horse_id):
        """
        Create a DataFrame with all the frames with annotations from a csv-file.
        :param horse_id: int
        :return: pd.DataFrame
        """
        df_csv = pd.read_csv('videos_overview_missingremoved.csv', sep=';')
        column_headers = ['Video_ID', 'Path', 'Pain', 'Observer', 'Train']
        horse_path = self.path + 'horse_' + str(horse_id) + '/'
        big_list = []
        for path, dirs, files in sorted(os.walk(horse_path)):
            print(path)
            for filename in sorted(files):
                total_path = join(path, filename)
                print(total_path)
                vid_id = get_video_id_stem_from_path(path)
                csv_row = df_csv.loc[df_csv['Video_id'] == vid_id]
                if '.jpg' in filename or '.png' in filename:
                    train_field = -1
                    pain_field = csv_row.iloc[0]['Pain']
                    observer_field = csv_row.iloc[0]['Observer']
                    row_list = [vid_id, total_path, pain_field,
                                observer_field, train_field]
                    big_list.append(row_list)
        horse_df = pd.DataFrame(big_list, columns=column_headers)
        return horse_df

    def save_OF_paths_to_df(self, horse_id, horse_df):
        """
        Create a DataFrame with all the optical flow paths with annotations
        from a csv-file, then join it with the existing horse df with rgb paths,
        at simultaneous frames.
        :param horse_id: int
        :param horse_df: pd.DataFrame
        :return: pd.DataFrame
        """
        # OF_path_df = pd.DataFrame(columns=['OF_Path'])  # Instantiate an empty df
        # of_header = ['OF_Path']
        c = 0  # Per horse frame counter.
        per_clip_frame_counter = 0
        old_path = 'NoPath'
        root_of_path = self.of_path + 'horse_' + str(horse_id) + '/'
        of_path_list = []

        # Walk through all the files in the of-folders and put them in a
        # DataFrame column, in order (the same order they were extracted in.)
        for path, dirs, files in sorted(os.walk(root_of_path)):
            print(path)
            video_id = get_video_id_from_path(path)
            nb_frames_in_clip = len(horse_df.loc[horse_df['Path'].str.contains(video_id)])
            print(video_id)
            if old_path != path and c != 0:  # If entering a new folder
                per_clip_frame_counter = 0
                if '1fps' in self.of_path: # To match the #pictures with #of I disregard the first frame.
                    horse_df.drop(c, inplace=True)  # Delete first element
                    horse_df.reset_index(drop=True, inplace=True)  # And adjust the index
            old_path = path
            for filename in sorted(files):
                total_path = join(path, filename)
                if '.npy' in filename or '.jpg' in filename:        # (If it's an optical flow-array.)
                    if per_clip_frame_counter > nb_frames_in_clip:  # This can probably be removed but will
                        break                                       # leave it here for now.
                    of_path_list.append(total_path)
                    c += 1
                    per_clip_frame_counter += 1

        # Now extend horse_df to contain both rgb and OF paths,
        # and then return whole thing.
        nb_of_paths = len(of_path_list)
        nb_rgb_frames = len(horse_df)
        if nb_rgb_frames != nb_of_paths:
            diff = nb_rgb_frames - nb_of_paths
            print("Differed by:", diff)
            # (They should only differ by one row.
            # Else an error should be raised when concatenating.)
            if diff < 0: # If the of-df was larger, reduce it
                of_path_list = of_path_list[:diff]
            else:  # Vice versa with horse-df
                horse_df = horse_df[:-diff]
        try:
            # Add column (concatenate)
            # horse_df.loc[:, 'OF_Path'] = pd.Series(OF_path_df['OF_Path'])
            horse_df.loc[:, 'OF_Path'] = pd.Series(of_path_list)
        except AssertionError:
            print('Horse df and OF_df were not the same length and could not'
                  'be concatenated. Even despite having removed the last'
                  'element of horse df which should be 1 longer.')

        return horse_df

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
        for path in df['Path']:
            im = process_image(path,
                               (self.image_size[0],
                                self.image_size[1],
                                channels))
            images.append(im)
        return images

def plot_augmentation(X_seq_list, flipped, cropped, shaded,
                      seq_index, batch_index, window_index):
    rows = 4
    cols = 10
    f, axarr = plt.subplots(rows, cols, figsize=(20,10))
    for i in range(0, rows):
        for j in range(0, cols):
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
    plt.savefig('seq_{}_batch_{}_wi_{}.png'.format(seq_index, batch_index, window_index))

def get_video_id_stem_from_path(path):
    _, vid_id = split_string_at_last_occurence_of_certain_char(path, '/')
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

