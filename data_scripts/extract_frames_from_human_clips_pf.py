import pandas as pd
import subprocess
import os


if __name__ == '__main__':
    fps_to_extract = 2
    # df = pd.read_csv('../data/pf/random_sequences/ground_truth_randomclips_lps.csv', sep=',')
    root_dir = '../data/pf/random_sequences/'
    frames_dir = '../data/pf/random_sequences/jpg_128_128_{}fps/'.format(fps_to_extract)

    if not os.path.exists(frames_dir):
        subprocess.call(['mkdir', frames_dir])

    for path, dirs, files in sorted(os.walk(root_dir)):
        # Create one folder per test clip
        for filename in sorted(files):
            if filename.endswith('.mp4'):
                print("NEW CLIP FROM STUDY, mkdir")
                output_dir_path = os.path.join(frames_dir, filename[:-4])
                if not os.path.exists(output_dir_path):
                    subprocess.call(['mkdir', output_dir_path])

    start = '00:00:00'  # Use whole videos here
    # Extract frames
    for path, dirs, files in sorted(os.walk(root_dir)):
        for filename in sorted(files):
            if filename.endswith('.mp4'):

                print("\nNEW CLIP FROM STUDY, extracting frames...")
                output_dir = filename[:-4]
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

