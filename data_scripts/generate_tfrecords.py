import tensorflow as tf


def int64_features(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = value.reshape(-1)
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def convert_to_sequential_example(feats, preds, labels, video_id, args, config_dict):
    """Build a SequenceExample proto for an example.
    Args:
        video_id: string, id for video file, e.g., '33467'
        video_buffer: numpy array with the video frames, with dims [n_frames, height, width, n_channels]
        label: integer, identifier for the ground truth class for the network
        args: args from argparse
    Returns:
        Example proto
    """
    nb_clips = config_dict['video_pad_length']
    assert feats.shape[0] == nb_clips
    assert feats.shape[1] == 20480
    assert preds.shape[0] == feats.shape[0]
    assert labels.shape[0] == feats.shape[0]

    features = {}
    features['nb_clips']    = int64_feature(feats.shape[0])
    features['height']      = int64_feature(1)
    features['width']       = int64_feature(feats.shape[1])
    features['features']    = bytes_feature(tf.io.serialize_tensor(feats))
    features['preds']       = bytes_feature(tf.io.serialize_tensor(preds))
    features['labels']      = bytes_feature(tf.io.serialize_tensor(labels))
    features['video_id']    = bytes_feature(str.encode(video_id))
  
    # Compress the frames using JPG and store in as a list of strings in 'frames'
  
    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example


