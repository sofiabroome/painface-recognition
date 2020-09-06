import pandas as pd
import subprocess
import argparse
import sys
import os

from lps_subjects import lps_subjects
sys.path.append('..')


def get_length(filename):
    result = subprocess.Popen(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    str_duration = result.stdout.readline()
    print(str_duration)
    str_duration = str_duration.rstrip() 
    return float(str_duration)

parser = argparse.ArgumentParser()
parser.add_argument('--dir', nargs='?', type=str,
                    help='The directory to list the videos from.')
parser.add_argument('--ext', nargs='?', type=str,
                    help='The file extension of the files'
                         'we want to list (for example .MP4')

args = parser.parse_args()

video_folder = args.dir
video_file_ext = args.ext

complete_paths = []
file_names = []

column_headers = ['subject', 'video_id', 'path', 'length', 'trial', 'observer', 'pain']
big_list = []


for dirpath, dirnames, files in os.walk(video_folder):
    if '.DS_Store' not in files[0]:
        for filename in files:
            fn_parts = filename.split('_')
            horse = lps_subjects[filename[0]]
            print(horse)
            complete_paths.append(os.path.join(dirpath, filename))
            file_names.append(filename)
            total_path = os.path.join(dirpath, filename)
            length = get_length(total_path)
            trial = fn_parts[2]
            video_id = filename[:-4]
            row_list = [horse, video_id, total_path, length, trial, 1, 0]
            big_list.append(row_list)

df = pd.DataFrame(big_list, columns=column_headers)
df = df.sort_values(by=['path'])
df = df.reset_index(drop=True)
df.to_csv('lps_videos_overview.csv')


