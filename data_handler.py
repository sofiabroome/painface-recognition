import _pickle as cp
import pandas as pd
import numpy as np
import random
import ipdb
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from os.path import join

from helpers import split_string_at_last_occurence_of_certain_char
from image_processor import process_image

datagen = ImageDataGenerator()

class DataHandler:
    def __init__(self, path, image_size, seq_length, batch_size, color, nb_labels):
        """
        Constructor for the DataHandler.
        :param path: str
        :param image_size: (int, int)
        :param seq_length: int
        :param color: bool
        """
        self.path = path
        self.image_size = image_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.color = color
        self.nb_labels = nb_labels

    def prepare_image_generators(self, df, train):
        """
        Prepare the frames into labeled train and test sets, with help from the
        DataFrame with .jpg-paths and labels for train and pain.
        :param df: pd.DataFrame
        :param train: Boolean
        :return: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """
        if train:
            df = df.loc[df['Train'] == 1]
        else:
            df = df.loc[df['Train'] == 0]
        print("LEN DF:")
        print(len(df))
        batch_index = 0

        for index, row in df.iterrows():
            if batch_index == 0:
                X_list = []
                y_list = []
            x = self.get_image(row['Path'])
            y = row['Pain']
            X_list.append(x)
            y_list.append(y)
            batch_index += 1

            if batch_index % self.batch_size == 0:
                X_array = np.array(X_list, dtype=np.float32)
                y_array = np.array(y_list, dtype=np.uint8)
                print("**************************************")
                print("Inside prep image generator:")
                print("X array shape:")
                print(X_array.shape)
                print("y array shape:")
                print(y_array.shape)
                y_array = np_utils.to_categorical(y_array, num_classes=self.nb_labels)
                print(y_array.shape)
                print("**************************************")
                # batch_index = 0
                X_array, y_array = datagen.flow(X_array, y_array, batch_size=self.batch_size).next()
                yield (X_array, y_array)

    def get_image(self, path):
        if self.color:
            channels = 3
        else:
            channels = 1
        im = process_image(path, (self.image_size[0], self.image_size[1], channels))
        return im

    ### BELOW HANDLES DATA EXTRACTION WHEN THE DATA IS ORGANIZED IN TRAIN/TEST/PAIN/NOPAIN-FOLDERS

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
        for path, dirs, files in os.walk(self.path):
            for filename in files:
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

    def horse_to_df(self, horse_id):
        """
        Create a DataFrame with all the frames with annotations from a csv-file.
        :return: pd.DataFrame
        """
        df_csv = pd.read_csv('videos_overview_missingremoved.csv', sep=';')
        horse_df = pd.DataFrame(columns=['Video_ID', 'Path', 'Pain', 'Observer', 'Train'])
        c = 0
        horse_path = self.path + 'horse_' + str(horse_id) + '/'
        for path, dirs, files in os.walk(horse_path):
            print(path)
            for filename in files:
                total_path = join(path, filename)
                vid_id = get_video_id_from_path(path)
                csv_row = df_csv.loc[df_csv['Video_id'] == vid_id]
                if '.jpg' in filename or '.png' in filename:
                    train_field = -1
                    pain_field = csv_row.iloc[0]['Pain']
                    observer_field = csv_row.iloc[0]['Observer']
                    horse_df.loc[c] = [vid_id, total_path, pain_field, observer_field, train_field]
                    c += 1
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
            im = process_image(path, (self.image_size[0], self.image_size[1], channels))
            images.append(im)
        return images


def get_video_id_from_path(path):
    _, vid_id = split_string_at_last_occurence_of_certain_char(path, '/')
    nb_underscore = vid_id.count('_')
    if nb_underscore > 1:
        vid_id, _ = split_string_at_last_occurence_of_certain_char(vid_id, '_')
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


def shuffle_blocks(df):
    vids = set(df['Video_ID'])
    df_blocks = []
    for v in vids:
        df_block = df[df['Video_ID'] == v]
        df_blocks.append(df_block)
    random.shuffle(df_blocks)
    df = pd.concat(df_blocks)
    return df
