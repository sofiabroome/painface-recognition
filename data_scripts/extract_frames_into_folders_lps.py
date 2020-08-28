import pandas as pd
import subprocess
import os

from lps_subjects import lps_subjects


if __name__ == '__main__':
    df = pd.read_csv('../metadata/lps_videos_overview.csv', sep=',')
    root_dir = '../data/lps/raw_videos/'
    frames_dir = '../data/lps/jpg_128_128_2fps/'

    if not os.path.exists(frames_dir):
        subprocess.call(['mkdir', frames_dir])

    # Make all the subfolders for all the separate 60 sequences, in separate horse_id folders.
    # NOTE: The horse_id folders need to be created beforehand, in the root data folder
    # like so: ./data/horse_id. Only need to do once. 

    for key, horse in lps_subjects.items():
        print("NEW HORSE")
        output_dir = horse
        output_dir_path = os.path.join(frames_dir, output_dir)
        if not os.path.exists(output_dir_path):
            subprocess.call(['mkdir', output_dir_path])
        horse_df = df.loc[df['subject'] == output_dir]
        for ind, row in horse_df.iterrows():
            path = row['path']
            seq_dir_path = os.path.join(frames_dir, output_dir, row['video_id'])
            subprocess.call(['mkdir', seq_dir_path])

    # Extract frames

    start = '00:00:00'  # Use whole videos here
    for key, horse in lps_subjects.items():
        print("NEW HORSE")
        output_dir = horse
        horse_df = df.loc[df['subject'] == output_dir]
        for ind, row in horse_df.iterrows():
            print(row['length'])
            seq_dir_path = os.path.join(frames_dir, output_dir, row['video_id'])

            # Start and lengths as hh:mm:ss-strings
            length = str(row['length'])
            print(start)

            complete_output_path = os.path.join(seq_dir_path, 'frame_%06d.jpg')
            video_path = str(row['path'])

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

            # CHOOSE ONE FROM THE FOLLOWING

            # JPG HALFASS QUALITY, MAYBE LOSSY, 16 FPS
            # NOTE:  Need to add qscale:v arg for higher frame rates, otherwise pixelated.
            # ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-qscale:v', str(4), '-t', length, '-vf',
            #                   'scale=320:240', '-r', str(16), '-an', complete_output_path]

            # JPG 2FPS 128x128
            #
            ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-qscale:v', str(4), '-t', length, '-vf',
                              'scale=128:128', '-r', str(2), '-an', complete_output_path]

            # JPG 16FPS 128x128
            
            # ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-qscale:v', str(4), '-t', length, '-vf',
            #                   'scale=128:128', '-r', str(16), '-an', complete_output_path]

            # TEST SETTINGS, JUST 3 FRAMES PER VIDEO:
            # ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-vcodec', 'png', '-t', '00:00:03', '-vf',
            #                   'scale=128:128', '-r', str(1), '-an', complete_output_path]

            print(ffmpeg_command)
            subprocess.call(ffmpeg_command)

