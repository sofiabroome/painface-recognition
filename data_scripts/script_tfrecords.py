import sys
sys.path.append('..')
import os
import helpers
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

import generate_tfrecords as gtfr


def process_files_and_write(df_summary, path_to_features, args, config_dict):

    default_array_str = 'arr_0'
    if args.dataset == 'lps':
        subj_codes = ['A', 'B', 'H', 'I', 'J', 'K', 'N', 'S']
    if args.dataset == 'pf':
        subj_codes = ['1', '2', '3', '4', '5', '6']

    for subj_code in subj_codes:
        if args.dataset == 'pf':
            df = df_summary[(df_summary.subject == int(subj_code))]
        if args.dataset == 'lps':
            df = df_summary[(df_summary.subject == subj_code)]

        output_filename = 'videofeats_132766best_flat_{}.tfrecords'.format(subj_code)
        output_file = os.path.join(args.output_folder, output_filename)
        print('Output file path: ', output_file)
        writer = tf.io.TFRecordWriter(output_file)

        for ind, row in df.iterrows():
            print('Index in df: ', ind, end='\r')
            video_id = str(row['video_id'])
            npz_path = path_to_features + video_id + '.npz'
            loaded = np.load(npz_path, allow_pickle=True)[default_array_str].tolist()
            feats = loaded['features'].astype(np.float32)
            f_shape = feats.shape
            preds = np.array(loaded['preds']).astype(np.float32)
            labels = np.array(loaded['labels']).astype(np.int32)

            example = gtfr.convert_to_sequential_example(feats,
                                                         preds,
                                                         labels,
                                                         video_id,
                                                         args,
                                                         config_dict)
            writer.write(example.SerializeToString())

        writer.close()
        

def main():
    parser = argparse.ArgumentParser(
        description='Some parameters.')
    parser.add_argument(
        '--output-folder', nargs='?', type=str,
        help='Folder where to output the tfrecords file')
    parser.add_argument(
        '--config', nargs='?', type=str,
        help='Config file')
    parser.add_argument(
        '--dataset', nargs='?', type=str,
        help='Config file')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    print('Saving results to %s' % args.output_folder)
    config_dict_module = helpers.load_module(args.config)
    config_dict = config_dict_module.config_dict

    feature_folder = config_dict['train_video_features_folder']
    path_to_features = config_dict['data_path'] + feature_folder
    df_summary = pd.read_csv(path_to_features + 'summary.csv')

    # Run it!
    process_files_and_write(df_summary, path_to_features, args, config_dict)


if __name__ == '__main__':
    main()
    print('\n')

