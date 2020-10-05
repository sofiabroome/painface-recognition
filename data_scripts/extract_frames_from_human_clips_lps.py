import pandas as pd
import subprocess
import os


if __name__ == '__main__':
    fps_to_extract = 16
    overview_df = pd.read_csv('../metadata/lps_videos_overview.csv', sep=',')
    df = pd.read_csv('../data/lps/random_clips_lps/ground_truth_randomclips_lps.csv', sep=',')
    root_dir = '../data/lps/random_clips_lps/'
    frames_dir = '../data/lps/random_clips_lps/jpg_128_128_{}fps/'.format(fps_to_extract)

    if not os.path.exists(frames_dir):
        subprocess.call(['mkdir', frames_dir])

    # Create one folder per test clip
    for index, row in df.iterrows():
        print("NEW CLIP FROM STUDY, mkdir")
        clip_index = row['ind']
        tc_x = 'test_clip_' + str(clip_index)
        output_dir_path = os.path.join(frames_dir, tc_x)
        if not os.path.exists(output_dir_path):
            subprocess.call(['mkdir', output_dir_path])

    # Extract frames

    start = '00:00:00'  # Use whole videos here
    for index, row in df.iterrows():
        print("\nNEW CLIP FROM STUDY, extracting frames...")
        clip_index = row['ind']
        tc_x = 'test_clip_' + str(clip_index)
        filename = tc_x + '.mp4' 
        output_dir = tc_x

        seq_dir_path = os.path.join(frames_dir, output_dir)

        # Start as hh:mm:ss-strings, length as number of seconds
        length = str(5)
        print(start)

        complete_output_path = os.path.join(seq_dir_path, 'frame_%06d.jpg')
        video_path = os.path.join(root_dir, filename)

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

        # JPG {}FPS 128x128

        ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-qscale:v', str(4), '-t', length, '-vf',
                          'scale=128:128', '-r', str(fps_to_extract), '-an', complete_output_path]

        print(ffmpeg_command)
        subprocess.call(ffmpeg_command)

