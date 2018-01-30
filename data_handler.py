import tensorflow as tf
import pandas as pd
import numpy as np
import random
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
                 seq_stride, batch_size, color, nb_labels):
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

        print("LEN DF (nb. of frames):")
        nb_frames = len(df)
        print(nb_frames)
        ws = self.seq_length  # "Window size" in a sliding window.
        ss = self.seq_stride  # Provide argument for slinding w. stride.
        valid = nb_frames - (ws - 1)
        nw = valid//ss  # Number of windows
        print('Number of windows', nw)
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
                    row = df.iloc[index]

                    x = self.get_image(row['Path'])
                    x /= 255  # Normalize to [0,1] since optical flow is on [0,1].
                    y = row['Pain']
                    flow = np.load(row['OF_Path'])
                    # Concatenate a third channel in order to comply w RGB images
                    # NOTE: If OF-path has 'magnitude' in it, no concatenation is needed and it already has 3 channels.
                    # Either just zeros, or the magnitude (can load magnitude directly now from file)
                    # extra_channel = np.zeros((flow.shape[0], flow.shape[1], 1))
                    # flow = np.concatenate((flow, extra_channel), axis=2)
                    X_seq_list.append(x)
                    y_seq_list.append(y)
                    flow_seq_list.append(flow)
                if batch_index == 0:
                    X_batch_list = []
                    y_batch_list = []
                    flow_batch_list = []
                X_batch_list.append(X_seq_list)
                y_batch_list.append(y_seq_list)
                flow_batch_list.append(flow_seq_list)
                batch_index += 1

                if batch_index % self.batch_size == 0 and not batch_index == 0:
                    X_array = np.array(X_batch_list, dtype=np.float32)
                    y_array = np.array(y_batch_list, dtype=np.uint8)
                    flow_array = np.array(flow_batch_list, dtype=np.float32)
                    if self.nb_labels == 2:
                        y_array = np_utils.to_categorical(y_array, num_classes=self.nb_labels)
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
        print("LEN DF:")
        print(nb_frames)
        ws = self.seq_length  # "Window size" in a sliding window.
        ss = self.seq_stride  # Provide argument for slinding w. stride.
        valid = nb_frames - (ws - 1)
        nw = valid//ss  # Number of windows
        print('Number of windows', nw)
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
                    row = df.iloc[index]
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
                X_seq_list = self.flip_images(X_seq_list)
                if batch_index == 0:
                    X_batch_list = []
                    y_batch_list = []
                X_batch_list.append(X_seq_list)
                y_batch_list.append(y_seq_list)
                batch_index += 1

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
        if self.color:
            channels = 3
        else:
            channels = 1
        im = process_image(path, (self.image_size[0], self.image_size[1], channels))
        return im

    def flip_images(self, images):
        X_flip = []
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, shape=(self.image_size[1], self.image_size[0], 3))
        tf_img1 = tf.image.flip_left_right(X)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for img in images:
                flipped_imgs = sess.run([tf_img1], feed_dict={X: img})
                X_flip.extend(flipped_imgs)
        X_flip = np.array(X_flip, dtype=np.float32)
        return X_flip

    # BELOW HANDLES DATA EXTRACTION WHEN THE DATA IS ORGANIZED IN TRAIN/TEST/PAIN/NOPAIN-FOLDERS

    def prepare_train_test(self, df):
        """
        Prepare the frames into labeled train and test sets, with help from the
        DataFrame with .jpg-paths and labels for train and pain.
        :param df: pd.DataFrame
        :return: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """
        train_df = df.loc[df['Train'] == 1]
        test_df = df.loc[df['Train'] == 0]

        y_train = np.asarray(train_df['Pain'], dtype=np.uint8)
        y_test = np.asarray(test_df['Pain'], dtype=np.uint8)

        print("**************************************")
        print("Inside prep traintest:")
        print(y_train.shape)
        print(y_test.shape)
        from keras.utils import np_utils
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        print(y_train.shape)
        print(y_test.shape)
        print("**************************************")

        print('Before get train ims')
        train_images_list = self._get_images_from_df(train_df)
        print('Before get test ims')
        test_images_list = self._get_images_from_df(test_df)
        print('Before convert train to array')
        X_train = np.asarray(train_images_list, dtype=np.float32)
        print('Before convert test to array')
        X_test = np.asarray(test_images_list, dtype=np.float32)

        # X_train_batch = make_even_sequences(train_images_list, self.seq_length)
        # y_train_batch = np.asarray(make_even_sequences(y_train, self.seq_length), dtype=np.uint8)
        # X_test_batch = make_even_sequences(test_images_list, self.seq_length)
        # y_test_batch = np.asarray(make_even_sequences(y_test, self.seq_length), dtype=np.uint8)
        #
        # X_train = X_train_batch
        # X_test = X_test_batch
        # y_train = y_train_batch
        # y_test = y_test_batch

        return X_train, y_train, X_test, y_test

    def folders_to_df(self):
        """
        Create a DataFrame with annotations from the folder structure.
        :return: pd.DataFrame
        """
        df = pd.DataFrame(columns=['FileName', 'Path', 'Pain', 'Observer', 'Train'])
        c = 0
        for path, dirs, files in sorted(os.walk(self.path)):
            for filename in sorted(files):
                total_path = join(path,filename)
                if '.jpg' in filename:
                    if 'train' in total_path:
                        train_field = 1
                    else:
                        train_field = 0
                    if 'pain' in total_path and 'no_pain' not in total_path:
                        pain_field = 1
                    else:
                        pain_field = 0
                    if 'observer' in total_path:
                        observer_field = 1
                    else:
                        observer_field = 0
                    df.loc[c] = [filename, total_path, pain_field, observer_field, train_field]
                    c += 1
        # A weird file .jpg appears in df, remove it.
        df = df[df['FileName'] != '.jpg']
        return df

    # TODO Merge the two below functions (horse_to_df and save_OF_paths_to_df, same functionality)
    def horse_to_df(self, horse_id):
        """
        Create a DataFrame with all the frames with annotations from a csv-file.
        :param horse_id: int
        :return: pd.DataFrame
        """
        df_csv = pd.read_csv('videos_overview_missingremoved.csv', sep=';')
        horse_df = pd.DataFrame(columns=['Video_ID',
                                         'Path',
                                         'Pain',
                                         'Observer',
                                         'Train'])
        c = 0
        horse_path = self.path + 'horse_' + str(horse_id) + '/'
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
                    horse_df.loc[c] = [vid_id,
                                       total_path,
                                       pain_field,
                                       observer_field,
                                       train_field]
                    c += 1
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
        OF_path_df = pd.DataFrame(columns=['OF_Path'])  # Instantiate an empty df
        c = 0  # Per horse frame counter.
        per_clip_frame_counter = 0
        old_path = 'NoPath'
        root_of_path = self.of_path + 'horse_' + str(horse_id) + '/'

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
                if '.npy' in filename:  # (If it's an optical flow-array.)
                    if per_clip_frame_counter > nb_frames_in_clip:  # This can probably be removed but will
                                                                    # leave it here for now.
                        break
                    OF_path_df.loc[c] = [total_path]
                    c += 1
                    per_clip_frame_counter += 1

        # Now extend horse_df to contain both rgb and OF paths,
        # and then return whole thing.

        if len(horse_df) != len(OF_path_df):
            diff = len(horse_df) - len(OF_path_df)
            print("Differed by:", diff)
            # (They should only differ by one row.
            # Else an error should be raised when concatenating.)
            if diff < 0: # If the of-df was larger, reduce it
                OF_path_df = OF_path_df[:diff]
            else:  # Vice versa with horse-df
                horse_df = horse_df[:-diff]

        try:
            # Add column (concatenate)
            horse_df.loc[:, 'OF_Path'] = pd.Series(OF_path_df['OF_Path'])
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

