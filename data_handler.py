import pandas as pd
import numpy as np
import os, csv

from os.path import join

from image_processor import process_image


class DataHandler:
    def __init__(self, path, image_size, color):
        """
        Constructor for the DataHandler.
        :param path: str
        :param image_size: (int, int)
        :param color: bool
        """
        self.path = path
        self.image_size = image_size
        self.color = color

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

        train_images_list = self._get_images_from_df(train_df)
        test_images_list = self._get_images_from_df(test_df)

        X_train = np.asarray(train_images_list, dtype=np.float32)
        X_test = np.asarray(test_images_list, dtype=np.float32)

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
                    if 'pain' in total_path and 'no pain' not in total_path:
                        pain_field = 1
                    else:
                        pain_field = 0
                    if 'observer' in total_path:
                        observer_field = 1
                    else:
                        observer_field = 0
                    df.loc[c] = [filename, total_path, pain_field, observer_field, train_field]
                    c += 1
        print(df)
        # A weird file .jpg appears in df, remove it.
        df = df[df['FileName'] != '.jpg']
        return df

    def folders_to_csv(self):
        with open(self.path + "/file_listing.csv", 'w') as f:
            writer = csv.writer(f)
            for path, dirs, files in os.walk(self.path):
                for filename in files:
                    if '.jpg' in filename:
                        writer.writerow([filename])

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
            print(path)
            im = process_image(path, (self.image_size[0], self.image_size[1], channels))
            images.append(im)
        return images
