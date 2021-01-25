import sys
sys.path.append('../') # Adds higher directory to python modules path.
from helpers import process_image, find_between
import pandas as pd
import subprocess
import argparse
import os

from lps_subjects import lps_subjects
import optical_flow

pd.set_option('max_colwidth', 800)

# Flow Options:
alpha = 0.012
ratio = 0.5625
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

parser = argparse.ArgumentParser(
    description='Python wrapper of Coarse2Fine Optical Flow')
parser.add_argument(
    '-viz', dest='viz', action='store_true',
    help='Visualize (i.e. save) output of flow.')
args = parser.parse_args()


def make_folders():
    """
    This method only needs to be run once. It creates the per-horse
    and per-sequence folders where to save the computed optical flow.
    :return: None
    """

    # Make all the subfolders for all the separate 60 sequences,
    # in separate horse_id folders.
    # The horse_id folders need to be created beforehand, once.

    for key, horse in lps_subjects.items():
        print("NEW HORSE")
        output_dir_path = os.path.join(output_root_dir, horse)
        if not os.path.exists(output_dir_path):
            subprocess.call(['mkdir', output_dir_path])
        horse_df = df.loc[df['subject'] == horse]
        for ind, row in horse_df.iterrows():
            seq_dir_path = os.path.join(output_dir_path, row['video_id'])
            subprocess.call(['mkdir', seq_dir_path])


def iterate_over_frames(frequency):

    for _, subject_id in lps_subjects.items():
        print(subject_id)
        csv_path = frames_dir + subject_id + '.csv'
        print(csv_path)
        subject_frames_df = pd.read_csv(csv_path, sep=',')
        counter = 0
        per_video_counter = 0
        # Every row in the df contains 1 video frame.
        for row in subject_frames_df.iterrows():
            if counter == 0:
                # The ims list will always contain maximum 2 images, between
                # which images the optical flow will be computed.
                ims = []
            if counter >= 2:
                if old_vid_seq_name != vid_seq_name:
                    print('New clip!', counter, per_video_counter)
                    print(vid_seq_name)
                    per_video_counter = 0
                    old_vid_seq_name = vid_seq_name
                # -1 if I want to start at 1, otherwise 2.
                counter_format = ("%06d" % (per_video_counter-1))
                frequency_counter = per_video_counter % frequency
                if frequency_counter == 2 or frequency_counter == 14:
                    flow_output_path_stem = output_root_dir + subject_id +  '/'\
                                            + vid_seq_name + '/flow_' + counter_format
                    # print(flow_output_path_stem)
                    optical_flow.compute_optical_flow(ims, flow_output_path_stem, magnitude=True)
                ims[0] = ims[1]
                ims.pop()
            frame_path = row[1]['path']
            # Equine data
            vid_seq_name = find_between(frame_path, subject_id + '/', '/frame')
            im = process_image('../' + frame_path, (width, height, channels),
                               standardize=False, mean=None, std=None, normalize=False)
            ims.append(im)
            counter += 1
            per_video_counter += 1
            if counter == 1:
                old_vid_seq_name = vid_seq_name


if __name__ == '__main__':
    # CSV with info about all the video sequences.
    df = pd.read_csv('../metadata/lps_videos_overview.csv')
    # df = pd.read_csv('../metadata/shoulder_pain_overview.csv')

    # Directory with the frames from which to extract the OF.
    frames_dir = '../data/lps/jpg_224_224_25fps/'

    # Output root directory (will contain subfolders for every sequence).
    # Need to make this folder before running, and the horse_x folders in it.
    output_root_dir = '../data/lps/jpg_224_224_25fps_OF_magnitude_cv2_2fpsrate/'
    if not os.path.exists(output_root_dir):
        subprocess.call(['mkdir', output_root_dir])

    width = 224
    height = 224
    channels = 3

    # Only need to make the subfolders of output_root_dir once.
    make_folders()

    # Iterate over the frames in root_dir and compute the flows.
    # Right now this is done for every 15th frame.
    iterate_over_frames(frequency=25)
