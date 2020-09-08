import pandas as pd
import subprocess
import random
import os

import extract_random_sequences


def get_random_start(video_row, sample_length):
    """ Get a random start time from a clip. Sample needs to fit within clip.
    param video_row: pd.DataFrame
    param sample_length: str on format hh:mm:ss
    return: str
    """
    start = pd.to_timedelta('00:00:00')
    end = pd.to_timedelta(video_row['length'], 's')
    valid_end = end - sample_length
    r = random.uniform(0.2, 0.8)
    randstart = start + r*(valid_end-start)
    return extract_random_sequences.format_timedelta(randstart)


if __name__ == '__main__':
    df = pd.read_csv('../data/selection_for_study.csv')

    root_dir = '../data/lps/raw_videos/'
    output_dir = '../data/random_clips_lps/'

    df_shuffled = df.sample(frac=1)

    clip_info = []
    column_headers = ['video_id', 'pain', 'ind']

    for ind, row in df_shuffled.iterrows():

        video_path = row['path']
        print('VIDEO PATH:')
        print(video_path)

        row_list = [row['video_id'], row['pain'], ind]
        clip_info.append(row_list)

        # Repeat three times to avoid humans in clip

        for i in range(5):

            # Start and lengths as hh:mm:ss-strings
            length = '00:00:05'
            start = get_random_start(row, length)
            print(start)

            name_elements = ['test_clip', str(ind), str(i)]
            complete_output_path = output_dir + '_'.join(name_elements) + '.mp4'

            print('COMPLETE OUTPUT PATH:')
            print(complete_output_path)

            if not os.path.exists(video_path):
                print("video path does not exist")

            # GOOD PNG QUALITY
            # Below command is used to extract part of a clip.
            # ffmpeg -ss [start] -i in.mp4 -t [duration] -c copy out.mp4
            ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-t',
                              length, '-c', 'copy', '-an', complete_output_path]

            print(ffmpeg_command)
            subprocess.call(ffmpeg_command)

    clip_info_df = pd.DataFrame(clip_info, columns=column_headers)
    clip_info_df.to_csv(path_or_buf='../data/random_clips_lps/ground_truth_randomclips_lps.csv')

