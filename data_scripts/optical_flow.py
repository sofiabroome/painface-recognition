import sys
sys.path.append('../') # Adds higher directory to python modules path.

from extract_frames_into_folders import check_if_unique_in_df
from helpers import process_image, find_between
import pandas as pd
import numpy as np
import subprocess
import imageio
import time
import argparse
import cv2
import os


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


def get_path(file_name, path_dict):
    """
    param file_name: str
    """
    return path_dict.get(file_name + '.mts')


def make_folders():
    """
    This method only needs to be run once. It creates the per-horse
    and per-sequence folders where to save the computed optical flow.
    :return: None
    """

    # Make all the subfolders for all the separate 60 sequences,
    # in separate horse_id folders.
    # The horse_id folders need to be created beforehand, once.

    subject_ids = subject_ID_df['Subject'].values

    for ind, subject_id in enumerate(subject_ids):
        print("NEW SUBJECT")
        counter = 1  # Counter of non-unique videos.
        output_dir = subject_id
        subject_df = df.loc[df['Subject'] == subject_id]
        for vid in subject_df['Video_ID']:
            occurences = check_if_unique_in_df(vid, df)
            if occurences == 1:
                seq_dir_path = output_root_dir + output_dir + '/' + vid
            elif occurences > 1:
                seq_dir_path = output_root_dir + output_dir + '/' + vid + '_' + str(counter)
                if counter == occurences:
                    counter = 1
                else:
                    counter += 1
            else:
                print('WARNING 0 occurences')
            subprocess.call(['mkdir', seq_dir_path])


def get_flow_magnitude(flow):
    """
    Compute the magnitude of the optical flow at every pixel.
    :param flow: np.ndarray [width, height, 2]
    :return: np.ndarray [width, height, 1]
    """
    rows = flow.shape[0]
    cols = flow.shape[1]
    magnitude = np.zeros((rows, cols, 1))
    for i in range(0, rows):
        for j in range(0, cols):
            xflow = flow[i, j, 0]
            yflow = flow[i, j, 1]
            mag = np.sqrt(np.power(xflow, 2) + np.power(yflow, 2))
            magnitude[i, j] = mag
    return magnitude


def compute_optical_flow(ims, output_path_stem, magnitude=True):
    im1, im2 = ims[0], ims[1]
    # im1 = im1.astype(float) / 255.
    # im2 = im2.astype(float) / 255.
    # s = time.time()
    # Compute the flow via Pyflow/Coarse2FineFlowWrapper.
    # https://github.com/pathak22/pyflow/blob/master/pyflow.pyx
    # u and are [h,w,2]-arrays (elements are float64 on [0,1])
    # Flow Options:
    # alpha = 0.012
    # ratio = 0.5625
    # minWidth = 20
    # nOuterFPIterations = 7
    # nInnerFPIterations = 1
    # nSORIterations = 30
    # colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    # Pyflow implementation:
    # u, v, im2W = pyflow.coarse2fine_flow(
    #     im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    #     nSORIterations, colType)
    # OpenCV G. Farneback dense implementation:
    # print(im1.shape, im2.shape)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(im1, im2, flow=None, pyr_scale=0.5, levels=3,
                                        winsize=15, iterations=3, poly_n=5,
                                        poly_sigma=1.2, flags=0)
    # e = time.time()
    # print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
    #     e - s, width, height, channels))
    # flow = np.concatenate((u[..., None], v[..., None]), axis=2)

    # ADD MAGNITUDE AS THIRD CHANNEL (optional)
    if magnitude:
        extra_channel = get_flow_magnitude(flow)
    else:
        rows = flow.shape[0]
        cols = flow.shape[1]
        extra_channel = np.zeros((rows, cols, 1))
    flow = np.concatenate((flow, extra_channel), axis=2)
    flow = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    imageio.imwrite(output_path_stem + '.jpg', flow)

    if args.viz:
        hsv = np.zeros(im1.shape, dtype=np.uint8)
        hsv[:, :, 0] = 255
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(output_path_stem + '.png', rgb)
        cv2.imwrite(output_path_stem + '.jpg', im2W[:, :, ::-1] * 255)


def iterate_over_frames(frequency):

    subject_ids = subject_ID_df['Subject'].values

    for subject_id in subject_ids:
        print(subject_id)
        csv_path = root_dir + subject_id + '.csv'
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
                counter_format = ("%06d" % (per_video_counter-1))  # -1 if I want to start at 1, otherwise 2.
                if (per_video_counter % frequency) == 2:
                    flow_output_path_stem = output_root_dir + subject_id +  '/'\
                                            + vid_seq_name + '/flow_' + counter_format
                    # print(flow_output_path_stem)
                    compute_optical_flow(ims, flow_output_path_stem, magnitude=True)
                # flow_output_path_stem = output_root_dir + subject_id +  '/'\
                #                         + vid_seq_name + '/flow_' + counter_format
                # print(flow_output_path_stem)
                # compute_optical_flow(ims, flow_output_path_stem, magnitude=False)
                ims[0] = ims[1]
                ims.pop()
            frame_path = row[1]['Path']
            # Shoulder pain
            # vid_seq_name = find_between(frame_path, subject_id + '/', '/')
            # Equine data
            vid_seq_name = find_between(frame_path, subject_id + '/', '/frame')
            im = process_image('../' + frame_path, (width, height, channels))
            ims.append(im)
            counter += 1
            per_video_counter += 1
            if counter == 1:
                old_vid_seq_name = vid_seq_name


if __name__ == '__main__':
    # CSV with info about all the video sequences.
    df = pd.read_csv('../metadata/videos_overview_missingremoved.csv', sep=';')
    # df = pd.read_csv('../metadata/shoulder_pain_overview.csv')

    # CSV with unique subject overview.
    subject_ID_df = pd.read_csv('../metadata/horse_subjects.csv')

    # subject_ID_df = pd.read_csv('../metadata/shoulder_pain_subjects.csv')
    
    # Directory with the frames from which to extract the OF.
    root_dir = '../data/jpg_320_240_16fps/'
    # root_dir = '../data/ShoulderPain_172x129/Images/'
    # Output root directory (will contain subfolders for every sequence).
    # Need to make this folder before running, and the horse_x folders in it.
    # output_root_dir = '../data/jpg_128_128_16fps_OF_1_flow_per_frame_cv2/'
    output_root_dir = '../data/jpg_320_240_16fps_OF_magnitude_cv2_2fpsrate/'
    # output_root_dir = '../data/ShoulderPain172x129_OF_cv2/'

    # width = 172
    # height = 129
    # channels = 3

    # width = 128
    # height = 128
    # channels = 3

    width = 320
    height = 240
    channels = 3

    # Only need to make the subfolders of output_root_dir once.
    make_folders()

    # Iterate over the frames in root_dir and compute the flows.
    # Right now this is done for every 15th frame.
    iterate_over_frames(frequency=8)
