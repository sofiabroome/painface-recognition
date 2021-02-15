import tensorflow as tf
import pandas as pd
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE


def gen(nb_samples, lengths, data, labels, sequence_labels=True):
    data = [x.astype(np.float32) for x in data]
    labels = [int(x) for x in labels]
    lengths = [int(x) for x in lengths]
    pad_length = data[0].shape[0]
    for i in range(nb_samples):
        x = data[i]
        label = labels[i]
        length = lengths[i]
        if sequence_labels:
            # One hot encoding of sequence labels
            label_seq = np.zeros((pad_length, 2))
            if label == 0:
                label_seq[:length, 0] = 1
            if label == 1:
                label_seq[:length, 1] = 1
            label = label_seq
        else:
            if label == 0:
                label = [1, 0]
            if label == 1:
                label = [0, 1]
        # Output example format if seq labels: (266,) (266, 2) 48
        yield x, label, length


def construct_dataset(nb_pain, nb_nopain, batch_size, config_dict, rng):

    pain, p_lengths = get_data(
        nb_pain,
        min_events=config_dict['min_events_pain'],
        max_events=config_dict['nb_events_pain'],
        max_event_length=config_dict['max_length_pain'],
        max_intensity=config_dict['max_intensity_pain'],
        base=config_dict['base_level'],
        T=config_dict['video_pad_length'],
        rng=rng)
    nopain, np_lengths = get_data(
        nb_nopain,
        min_events=config_dict['min_events_nopain'],
        max_events=config_dict['nb_events_nopain'],
        max_event_length=config_dict['max_length_nopain'],
        max_intensity=config_dict['max_intensity_nopain'],
        base=config_dict['base_level'],
        T=config_dict['video_pad_length'],
        rng=rng)
    print('first 5 pain seq lengths: ', p_lengths[:5])
    print('first 5 nopain seq lengths: ', np_lengths[:5])
    data = nopain + pain
    # Standardize
    data = (data - np.mean(data))/np.std(data)
    # values /= sum(values)
    labels = np.zeros(nb_nopain).tolist() + np.ones(nb_pain).tolist()
    dataset = tf.data.Dataset.from_generator(lambda: gen(nb_pain+nb_nopain, np_lengths + p_lengths, data, labels),
                                             output_types=(tf.float32, tf.int32, tf.int32))
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def get_data(nb_series, min_events, max_events, max_event_length, max_intensity, base, T, rng):
    series = []
    series_lengths = []
    lengths = pd.read_csv('../metadata/video_lengths.csv')[51:]
    mu = lengths['length'].mean()
    sigma = lengths['length'].std()

    for i in range(nb_series):
        length_draw = int(rng.normal(mu, sigma))
        series_lengths.append(length_draw)
        values = rng.normal(size=T)*base
        values = abs(values)
        values[length_draw:] = 0
        nb_events = 0 if max_events == 0 else rng.integers(min_events, max_events)
        for ev in range(nb_events):
            length = rng.integers(1, max_event_length+1)
            last_valid_start = length_draw - max_event_length
            start = 0 if last_valid_start == 0 else rng.integers(int(last_valid_start))
            end = start + length
            event = np.zeros(int(T))
            event[start:end] = values[range(start, end, 1)]*max_intensity
            values += event
        series.append(values)
    return series, series_lengths
