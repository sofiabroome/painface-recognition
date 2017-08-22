import os, csv
import pandas as pd

from os.path import join


class DataHandler:
    def __init__(self, path, image_size):
        self.path = path
        self.image_size = image_size

    def prepare_train_test(self):
        pass

    def folders_to_df(self):
        df = pd.DataFrame(columns=['FileName', 'Pain', 'Observer', 'Train'])
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
                    df.loc[c] = [filename, pain_field, observer_field, train_field]
                    c += 1
        print(df)
        return df

    def folders_to_csv(self):
        with open(self.path + "/file_listing.csv", 'w') as f:
            writer = csv.writer(f)
            # field_names = ['file_name', 'dirname']
            # writer = csv.DictWriter(f, fieldnames=field_names)
            # writer.writeheader()
            for path, dirs, files in os.walk(self.path):
                for d in dirs:
                    for filename in files:
                        if '.jpg' in filename:
                            writer.writerow([filename])
