import os, csv


class DataHandler:
    def __init__(self, path, image_size):
        self.path = path
        self.image_size = image_size

    def prepare_train_test(self):
        pass

    def folders_to_csv(self):
        with open(self.path + "/file_listing.csv", 'w') as f:
            writer = csv.writer(f)
            for path, dirs, files in os.walk(self.path):
                for filename in files:
                    writer.writerow([filename])
