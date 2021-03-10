import sys
sys.path.append('..')
import pandas as pd
import data_handler
import helpers
import os


def all_frames_in_folder_to_df(frames_path, df_csv, data_columns):
    """
    Create a DataFrame with all the frames with annotations from a csv-file.
    :param frames_path: str
    :param clip_csv: str
    :return: pd.DataFrame
    """
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


def add_flow_to_frames_df(flow_path, frames_df):
    """
    Create a DataFrame with all the optical flow paths with annotations
    from a csv-file, then join it with the existing df with rgb paths,
    at simultaneous frames.
    :param flow_path:
    :param frames_df:
    :return: pd.DataFrame
    """
    c = 0  # Per subject frame counter.
    per_clip_frame_counter = 0
    old_path = 'NoPath'
    of_path_list = []
    list_of_video_ids = list(set(frames_df['video_id'].values))

    # Walk through all the files in the of-folders and put them in a
    # list, in order (the same order they were extracted in.)

    for path, dirs, files in sorted(os.walk(flow_path)):
        print(path)
        if 'test_clip_' not in path:
            continue
        _, test_clip_x = path.rsplit(sep='/', maxsplit=1)
        print('test_clip_x: ', test_clip_x)
        _, clip_index = test_clip_x.rsplit(sep='_', maxsplit=1)
        csv_row = df_clip_csv.loc[df_clip_csv['ind'] == int(clip_index)]
        video_id = csv_row.iloc[0]['video_id']
        if video_id not in list_of_video_ids:
            print(video_id, ' was excluded')
            continue
        print(video_id)
        nb_frames_in_clip = len(
            frames_df.loc[frames_df['video_id'] == video_id])
        if old_path != path and c != 0:  # If entering a new folder (but not first time)
            per_clip_frame_counter = 0
            if '1fps' in flow_path:
                print('Dropping first optical flow to match with rgb.')
                frames_df.drop(c, inplace=True)  # Delete first element
                frames_df.reset_index(drop=True, inplace=True)  # And adjust the index
        old_path = path
        for filename in sorted(files):
            total_path = os.path.join(path, filename)
            if filename.startswith('flow_')\
                    and ('.npy' in filename or '.jpg' in filename):
                print(total_path)
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
                frames_df.drop(c, inplace=True)
                frames_df.reset_index(drop=True, inplace=True)
                print('Dropped the last rgb frame of the clip. \n')

    # Now extend subject_df to contain both rgb and OF paths,
    # and then return whole thing.
    try:
        frames_df.loc[:, 'of_path'] = pd.Series(of_path_list)
        frames_df.loc[:, 'train'] = -1
    except AssertionError:
        print('RGB and flow columns were not the same length'
              'and the data could not be merged.')

    return frames_df


def get_dataset_from_df(df, data_columns, config_dict, all_subjects_df):
    dh = data_handler.DataHandler(data_columns, config_dict, all_subjects_df)
    sequences_df = dh.get_sequences_from_frame_df(df)
    nb_steps_assuming_bs1 = len(sequences_df)
    print('Number of extracted sequences: ', len(sequences_df))
    # Set train to True to shuffle sequences
    return dh.get_dataset(sequence_dfs=sequences_df, train=True), nb_steps_assuming_bs1


def make_df_frames_rgb():
    if os.path.isfile(frames_csv_path):
        df_frames = pd.read_csv(frames_csv_path)
    else:
        print('Making a DataFrame with RGB frames for: ', frames_path)
        df_frames = all_frames_in_folder_to_df(frames_path,
                                               df_csv=df_clip_csv,
                                               data_columns=data_columns)
        df_frames.to_csv(frames_path + 'test_clip_frames.csv')
    return df_frames


def make_df_frames_optical_flow():
    if os.path.isfile(flow_frames_csv_path):
        df_flow_and_frames = pd.read_csv(flow_frames_csv_path)
    else:
        print('Making a DataFrame with optical flow frames for: ', frames_path)
        frames_df = make_df_frames_rgb()
        df_flow_and_frames = add_flow_to_frames_df(flow_path=flow_frames_path,
                                                   frames_df=frames_df)
        df_flow_and_frames.to_csv(flow_frames_path + 'test_clip_frames.csv')
        # df_flow_and_frames.to_csv(str_csv)
    return df_flow_and_frames

if __name__ == '__main__':

    # str_csv = 'data/lps/interpretability_results/top_k/A_20190104_IND3_STA_2/top_3_pain.csv'
    all_subjects_df = pd.read_csv('../metadata/horse_subjects.csv')
    data_columns = ['pain', 'observer']

    clip_csv_path = '../data/lps/random_clips_lps/ground_truth_randomclips_lps.csv'
    df_clip_csv = pd.read_csv(clip_csv_path)

    print('just tests')
    fps = 2
    frames_path = '../data/lps/random_clips_lps/jpg_224_224_{}fps/'.format(fps)
    flow_frames_path = '../data/lps/random_clips_lps/jpg_224_224_25fps_OF_magnitude_{}fpsrate/'.format(fps)

    frames_csv_path = frames_path + 'test_clip_frames.csv'
    flow_frames_csv_path = flow_frames_path + 'test_clip_frames.csv'

    config_dict_module = helpers.load_module('../configs/config_interpretability.py')
    config_dict = config_dict_module.config_dict

    # df_frames = make_df_frames_rgb()
    df_frames_flow = make_df_frames_optical_flow()
