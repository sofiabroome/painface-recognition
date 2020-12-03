import cv2
import math
import numpy as np
import tensorflow as tf

from helpers import util


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def convert_to_sequential_example(video_id, video_buffer, label, args, nb_frames=None):
    """Build a SequenceExample proto for an example.
    Args:
        video_id: string, id for video file, e.g., '33467'
        video_buffer: numpy array with the video frames, with dims [n_frames, height, width, n_channels]
        label: integer, identifier for the ground truth class for the network
        args: args from argparse
    Returns:
        Example proto
    """
    nb_frames = args.nb_frames if nb_frames is None else nb_frames
    assert len(video_buffer) == nb_frames
    assert video_buffer.shape[1] == args.height
    assert video_buffer.shape[2] == args.width

    features = {}
    features['nb_frames']   = int64_feature(video_buffer.shape[0])
    features['height']      = int64_feature(video_buffer.shape[1])
    features['width']       = int64_feature(video_buffer.shape[2])
    features['label']       = int64_feature(label)
    features['video_id']    = bytes_feature(str.encode(video_id))
  
    # Compress the frames using JPG and store in as a list of strings in 'frames'
    encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[1].tobytes())
                      for frame in video_buffer]
    features['frames'] = bytes_list_feature(encoded_frames)
  
    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example


def get_video_buffer(frame_folder_path, start_frame, end_frame):
    """Build a np.array of all frames in a sequence.
    Args:
        frames_folder_path: string, the path to the folder containing the frames.
        nb_frames: int, number of frames in sequence.
    Returns:
        frames: np.array [nb_frames, height, width, 3]
    """
    images = []
    for f in range(start_frame, end_frame+1):
        counter_format = ("%02d" % (f))  # 01 etc.
        # folder/frame01.jpg
        frame_path = get_frame_path(frame_folder_path, counter_format) 
        im = util.process_image(frame_path)
        images.append(im)
    frames = np.asarray(images)
    return frames

def get_frame_path(frame_folder_path, counter_format)
    return '{}frame{}.jpg'.format(frame_folder_path, counter_format)

def get_cohesive_crop_of_frames_video_buffer(frame_folder_path, start_frame, end_frame, nb_frames):
    """Build a np.array of a sampled, cohesive crop of x frames from a sequence of frames.
    Args:
        frames_folder_path: string, the path to the folder containing the frames.
        start_frame: int, index of first frame.
        end_frame: int, index of last frame.
        nb_frames: int, number of frames in sequence.
    Returns:
        frames: np.array [nb_frames, height, width, 3]
    """
    print('Sampling {} cohesive frames from every clip...'.format(nb_frames)) 
    print('Start frame index: {}, end frame index: {}'.format(start_frame,
                                                              end_frame)) 
    print('\n')
    images = []
    total_frames= (end_frame-start_frame)
    # If the clip has fewer frames than we want to sample
    if total_frames < nb_frames:
        assert total_frames > 0
        # Just return all frames
        sampled_frames = get_list_of_sampled_frames(start_frame,
                                                    end_frame,
                                                    total_frames)
        # Repeat the last frame for the remaining number of frames
        last_frame =  sampled_frames[-1]
        diff = nb_frames - total_frames
        for i in range(diff):
            sampled_frames.append(last_frame)
    else:
        sampled_frames = get_list_of_cohesive_frames(start_frame,
                                                     end_frame,
                                                     nb_frames)

    for f in sampled_frames:
        counter_format = ("%02d" % (f))  # 01 etc. This is how the frames were saved earlier.
        frame_path = get_frame_path(frame_folder_path, counter_format)  # folder/frame01.jpg
        print(frame_path)
        im = util.process_image(frame_path)
        images.append(im)

    assert len(images) == nb_frames
    frames = np.asarray(images)

    return frames


def get_fixed_number_of_frames_video_buffer(frame_folder_path, start_frame, end_frame, nb_frames):
    """Build a np.array of sampled frames from a sequence.
    Args:
        frames_folder_path: string, the path to the folder containing the frames.
        nb_frames: int, number of frames in sequence.
    Returns:
        frames: np.array [nb_frames, height, width, 3]
    """
    print('Sampling {} frames from every clip...'.format(nb_frames)) 
    print('\n')
    images = []
    total_frames= (end_frame-start_frame)
    # If the clip has fewer frames than we want to sample
    if total_frames < nb_frames:
        assert total_frames > 0
        # Just return all frames
        sampled_frames = get_list_of_sampled_frames(start_frame,
                                                    end_frame,
                                                    total_frames)
        # Repeat the last frame for the remaining number of frames
        last_frame =  sampled_frames[-1]
        diff = nb_frames - total_frames
        for i in range(diff):
            sampled_frames.append(last_frame)
    else:
        sampled_frames = get_list_of_sampled_frames(start_frame,
                                                    end_frame,
                                                    nb_frames)

    for f in sampled_frames:
        counter_format = ("%02d" % (f))  # 01 etc.
        # frame_path example: folder/frame01.jpg
        frame_path = get_frame_path(frame_folder_path, counter_format)  
        print(frame_path)
        im = util.process_image(frame_path)
        images.append(im)

    assert len(images) == nb_frames
    frames = np.asarray(images)

    return frames


def get_list_of_sampled_frames(start_frame, end_frame, nb_frames_to_sample):
    frames = range(start_frame, end_frame+1)
    length = float(len(frames))
    sampled_frames = []
    for i in range(nb_frames_to_sample):
        sampled_frames.append(frames[int(math.ceil(i * length / nb_frames_to_sample))])
    return sampled_frames


def get_list_of_cohesive_frames(start_frame, end_frame, nb_frames_to_sample):
    frames = list(range(start_frame, end_frame+1))
    return frames

