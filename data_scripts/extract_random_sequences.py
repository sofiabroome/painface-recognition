import pandas as pd
import subprocess
import datetime
import random
import os


def make_delta(entry):
    h, m, s = entry.split(':')
    return datetime.timedelta(hours=int(h),
                              minutes=int(m),
                              seconds=int(s))


def get_path(file_name):
    """
    param file_name: str
    """
    return path_dict.get(file_name + '.mts')


def check_if_unique_in_df(file_name, df):
    """
    param file_name: str
    param df: pd.DataFrame
    :return: int [nb occurences of sequences from the same video clip]
    """
    return len(df[df['Video_ID'] == file_name])


def format_timedelta(td):
    minutes, seconds = divmod(td.seconds + td.days * 86400, 60)
    hours, minutes = divmod(minutes, 60)
    return '{:d}:{:02d}:{:02d}'.format(hours, minutes, seconds)


def get_random_start(video_row, sample_length):
    """ Get a random start time from a clip. Sample needs to fit within clip.
    param video_row: pd.DataFrame
    param sample_length: str on format hh:mm:ss
    """
    start = pd.to_timedelta(video_row['Start'])
    end = pd.to_timedelta(video_row['End'])
    valid_end = end - sample_length
    r = random.uniform(0, 1)
    randstart = start + r*(valid_end-start)
    return format_timedelta(randstart)


if __name__ == '__main__':
    df = pd.read_csv('../metadata/videos_overview_missingremoved.csv', sep=';')
    root_dir = '../data/Experimental_pain/'
    complete_paths = []
    file_names = []
    filename = -1

    for dirpath, dirnames, files in os.walk(root_dir):
        if '.DS_Store' not in files[0]:
            for filename in files:
                complete_paths.append(os.path.join(dirpath, filename))
                file_names.append(filename)

    path_dict = dict((fn, p) for fn, p in zip(file_names, complete_paths))

    clip_info = []
    column_headers = ['Path', 'Video_ID', 'Pain', 'Start']

    output_dir = '../data/random_sequences_extra_videos/'
    lo = 1000
    hi = 1600
    random_ints = list(set([int(random.uniform(lo, hi)) for i in range(500)]))
    random.shuffle(random_ints)

    # Extract clips

    for h in range(1, 7):

        print("NEW HORSE")
        horse_df = df.loc[df['Subject'] == h]

        for ind, row in horse_df.iterrows():
            video_length = row['Length']
            pain = row['Pain']
            print(video_length)

            video_ID = row['Video_ID']
            video_path = str(get_path(video_ID))
            print('VIDEO PATH:')
            print(video_path)

            # Start and lengths as hh:mm:ss-strings
            length = '00:00:05'
            start = get_random_start(row, length)
            print(start)

            complete_output_path = output_dir + str(random_ints.pop()) + '.mp4'

            row_list = [complete_output_path, video_ID, pain, start]
            clip_info.append(row_list)

            print('COMPLETE OUTPUT PATH:')
            print(complete_output_path)

            if not os.path.exists(video_path):
                print("video path does not exist")

            # GOOD PNG QUALITY
            # Below command is used to extract part of a clip.
            # ffmpeg -ss [start] -i in.mp4 -t [duration] -c copy out.mp4
            ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-t', length, '-c', 'copy', '-an', complete_output_path]

            print(ffmpeg_command)
            subprocess.call(ffmpeg_command)

    clip_info_df = pd.DataFrame(clip_info, columns=column_headers)
    clip_info_df.to_csv(path_or_buf='../data/random_sequences_extra_videos/clip_info.csv')

