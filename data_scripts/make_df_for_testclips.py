import sys
sys.path.append('..')
import pandas as pd
import data_handler
import helpers
import os


def all_frames_in_folder_to_df(frames_path, clip_csv, data_columns):
    """
    Create a DataFrame with all the frames with annotations from a csv-file.
    :param frames_path: str
    :param clip_csv: str
    :return: pd.DataFrame
    """
    df_csv = pd.read_csv(clip_csv)
    column_headers = ['video_id', 'path', 'train']
    for dc in data_columns:
        column_headers.append(dc)
    print(column_headers)
    big_list = []
    for frames_path, dirs, files in sorted(os.walk(frames_path)):
        print(frames_path)
        for filename in sorted(files):
            if filename.startswith('frame_')\
                    and ('.jpg' in filename or '.png' in filename):
                total_path = os.path.join(frames_path, filename)
                print(total_path)
                tc_id = frames_path.rsplit(sep='/', maxsplit=1)[1]
                clip_index = tc_id.rsplit(sep='_', maxsplit=1)[1]
                csv_row = df_csv.loc[df_csv['ind'] == int(clip_index)]
                vid_id = csv_row.iloc[0]['video_id']
                if csv_row.empty:
                    continue
                train_field = -1
                row_list = [vid_id, total_path, train_field]
                for dc in data_columns:
                    if dc == 'pain':
                        field = csv_row.iloc[0][dc]
                        pain = 1 if field > 0 else 0
                        field = pain
                    if dc == 'observer':
                        field = 1
                    row_list.append(field)
                big_list.append(row_list)
    df = pd.DataFrame(big_list, columns=column_headers)
    return df


def get_dataset_from_df(df, data_columns, config_dict, all_subjects_df):
    dh = data_handler.DataHandler(data_columns, config_dict, all_subjects_df)
    return dh.get_dataset(df, train=False)
    

if __name__ == '__main__':
    print('just tests')
    # frames_path = '../data/lps/random_clips_lps/jpg_128_128_2fps/'
    # data_columns = ['pain', 'observer']
    # config_dict_module = helpers.load_module('../configs/config_interpretability.py')
    # config_dict = config_dict_module.config_dict
    # all_subjects_df = pd.read_csv('../metadata/horse_subjects.csv')
    # df = all_frames_in_folder_to_df(
    #         frames_path=frames_path,
    #         clip_csv='../data/lps/random_clips_lps/ground_truth_randomclips_lps.csv',
    #         data_columns=data_columns)
    # # df.to_csv(frames_path + 'test_clip_frames.csv')
