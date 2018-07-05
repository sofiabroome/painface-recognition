import pandas as pd
import subprocess
import datetime
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


if __name__ == '__main__':
    df = pd.read_csv('videos_overview_missingremoved.csv', sep=';')
    root_dir = 'data/Experimental_pain/'
    complete_paths = []
    file_names = []
    filename = -1

    for dirpath, dirnames, files in os.walk(root_dir):
        if '.DS_Store' not in files[0]:
            for filename in files:
                complete_paths.append(os.path.join(dirpath, filename))
                file_names.append(filename)

    path_dict = dict((fn, p) for fn, p in zip(file_names, complete_paths))

    # Make all the subfolders for all the separate 60 sequences, in separate horse_id folders.
    # NOTE: The horse_id folders need to be created beforehand, in the root data folder
    # like so: ./data/horse_id. Only need to do once. 

    for h in range(1, 7):
        print("NEW HORSE")
        counter = 1  # Counter of non-unique videos.
        output_dir = 'horse_' + str(h)
        horse_df = df.loc[df['Subject'] == h]
        for vid in horse_df['Video_ID']:
            path = get_path(vid)
            occurences = check_if_unique_in_df(vid, df)
            if occurences == 1:
                seq_dir_path = 'data/' + output_dir + '/' + vid
            elif occurences > 1:
                seq_dir_path = 'data/' + output_dir + '/' + vid + '_' + str(counter)
                if counter == occurences:
                    counter = 1
                else:
                    counter += 1
            else:
                print('WARNING 0 occurences')
            subprocess.call(['mkdir', seq_dir_path])

    # Extract frames
    for h in range(1, 7):
        print("NEW HORSE")
        counter = 1  # Counter of non-unique videos.
        output_dir = 'horse_' + str(h)
        horse_df = df.loc[df['Subject'] == h]
        for vid in horse_df.iterrows():
            # vid is a tuple, vid[0] is just the index,
            # vid[1] is the actual row.
            print(vid[1]['Length'])
            occurences = check_if_unique_in_df(vid[1]['Video_ID'], df)
            print(occurences)
            if occurences == 1:
                seq_dir_path = 'data/' + output_dir + '/' + vid[1]['Video_ID']
            elif occurences > 1:
                seq_dir_path = 'data/' + output_dir + '/' + vid[1]['Video_ID'] + '_' + str(counter)
                if counter == occurences:
                    counter = 1
                else:
                    counter += 1
            else:
                print("WARNING, No occurences")

            # Start and lengths as hh:mm:ss-strings
            start = str(vid[1]['Start'])
            length = str(vid[1]['Length'])
            print(start)

            # REAL
            # complete_output_path = '~/Documents/EquineML/painface-recognition/' + seq_dir_path + '/frame_%06d.jpg'

            # TEST
            # complete_output_path = 'data_test/frame_%06d.png'
            # video_path = '~/Documents/EquineML/painface-recognition/' + str(get_path(vid[1]['Video_ID']))

            complete_output_path = seq_dir_path + '/frame_%06d.jpg'
            video_path = str(get_path(vid[1]['Video_ID']))

            print('COMPLETE OUTPUT PATH:')
            print(complete_output_path)
            print('VIDEO PATH:')
            print(video_path)

            if not os.path.exists(complete_output_path):
                print("comp path does not exist")

            if not os.path.exists(seq_dir_path):
                print("seq path does not exist")

            if not os.path.exists(video_path):
                print("video path does not exist")

            print(os.environ['PATH'])

            # CHOOSE FROM THE FOLLOWING

            # GOOD PNG QUALITY, 5 FPS
            # complete_output_path = seq_dir_path + '/frame_%06d.png'
            # ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-vcodec', 'png', '-t', length, '-vf',
            #                   'scale=320:240', '-r', str(5), '-an', complete_output_path]

            # # JPG HALFASS QUALITY, MAYBE LOSSY, 1 FPS
            #
            #
            # ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-t', length, '-vf',
            #                   'scale=320:240', '-r', str(1), '-an', complete_output_path]

            # # JPG HALFASS QUALITY, MAYBE LOSSY, 2 FPS
            #
            #
            ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-t', length, '-vf',
                              'scale=320:240', '-r', str(2), '-an', complete_output_path]

            # JPG HALFASS QUALITY, MAYBE LOSSY, 15 FPS
            # NOTE:  Need to add qscale:v arg for higher frame rates, otherwise pixelated.
            # ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-qscale:v', str(4), '-t', length, '-vf',
            #                   'scale=320:240', '-r', str(15), '-an', complete_output_path]

            #JPG 2FPS 128x128
            #
            # ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-qscale:v', str(4), '-t', length, '-vf',
            #                   'scale=128:128', '-r', str(2), '-an', complete_output_path]

            # #JPG 15FPS 128x128
            #
            # ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-qscale:v', str(4), '-t', length, '-vf',
            #                   'scale=128:128', '-r', str(15), '-an', complete_output_path]

            #JPG 16FPS 128x128
            
            # ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-qscale:v', str(4), '-t', length, '-vf',
            #                   'scale=128:128', '-r', str(16), '-an', complete_output_path]

            # #JPG 1FPS 128x128

            # ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-qscale:v', str(4), '-t', length, '-vf',
            #                   'scale=128:128', '-r', str(1), '-an', complete_output_path]


            # TEST SETTINGS, JUST 3 FRAMES PER VIDEO:
            # ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-vcodec', 'png', '-t', '00:00:03', '-vf',
            #                   'scale=128:128', '-r', str(1), '-an', complete_output_path]

            print(ffmpeg_command)
            subprocess.call(ffmpeg_command)





