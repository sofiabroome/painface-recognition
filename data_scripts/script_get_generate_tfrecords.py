import os
import argparse

import pandas as pd
import tensorflow as tf

import generate_tfrecords as gtfr


def process_files_and_write(df, labels_df, split_folder, args):

    output_filename = split_folder[:-1] + '.tfrecords'
    output_file = os.path.join(args.output_folder, output_filename)
    print('Output file path: ', output_file)
    writer = tf.python_io.TFRecordWriter(output_file)

    for ind, row in df.iterrows():
        print('Index in df: ', ind, end='\r')
        video_id = str(row['id'])
        label = row['template'].replace('[','').replace(']','')
        label_number = labels_df[label]
        label_folder_path = split_folder + str(label_number) + '/' 
        frames_folder_path = label_folder_path + video_id + '/'
        video_buffer = gtfr.get_video_buffer(frames_folder_path,
                                             start_frame=0,
                                             end_frame=args.nb_frames)
        example = gtfr.convert_to_sequential_example(video_id,
                                                     video_buffer,
                                                     label_number,
                                                     args)
        writer.write(example.SerializeToString())

    writer.close()
        

def main():
    train_json = '~/Downloads/something-something-v2-train.json'
    val_json = '~/Downloads/something-something-v2-validation.json'
    
    test_json = '~/Downloads/something-something-v2-test.json'
    
    labels_json = '~/Downloads/something-something-v2-labels.json'
    
    labels_df = pd.read_json(labels_json, typ='series')

    # SET VAL OR TRAIN

    train_df = pd.read_json(train_json)
    split_folder = 'train_128/'

    # val_df = pd.read_json(val_json)
    # split_folder = 'validation_128/'

    parser = argparse.ArgumentParser(
        description='Some parameters.')
    parser.add_argument(
        '--output-folder', nargs='?', type=str,
        help='Folder where to output the tfrecords file')
    parser.add_argument(
        '--nb-frames', nargs='?', type=int,
        help='The number of frames per sequence.')
    parser.add_argument(
        '--width', nargs='?', type=int,
        help='Image width')
    parser.add_argument(
        '--height', nargs='?', type=int,
        help='Image height')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    print('Saving results to %s' % args.output_folder)

    # Run it!
    process_files_and_write(train_df, labels_df, split_folder, args)


if __name__ == '__main__':
    main()
    print('\n')

