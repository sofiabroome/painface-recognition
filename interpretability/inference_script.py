import sys
sys.path.append('..')

import tensorflow as tf
import pandas as pd
import numpy as np
import arg_parser
import helpers
import models
import wandb
import os

from interpretability import mask, gradcam as gc, interpretability_viz as viz
from data_handler import get_video_id_from_frame_path
from data_scripts import make_df_for_testclips


def run():

    if not os.path.exists(config_dict['output_folder']):
        os.makedirs(config_dict['output_folder'])

    model = models.MyModel(config_dict=config_dict).model
    model.load_weights(config_dict['checkpoint']).expect_partial()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config_dict['lr'],
        decay_steps=config_dict['lr_decay_steps'],
        decay_rate=0.96,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule)

    # gt_human_clips_lps_path = '/Midgard/Data/sbroome/painface-recognition/lps/random_clips_lps/ground_truth_randomclips_lps.csv'
    gt_human_clips_lps_path = '../data/lps/random_clips_lps/ground_truth_randomclips_lps.csv'
    gt_human_clips_df = pd.read_csv(gt_human_clips_lps_path)

    results_list = []
    col_headers = ['test_clip_id', 'video_id', 'cps_score', 'label', 'prediction', 'confidence']

    results_folder = os.path.join(config_dict['output_folder'], str(config_dict['job_identifier']))

    for sample_ind, sample in enumerate(dataset):
        print(sample_ind)
        if sample_ind == nb_steps_assuming_bs1:
            break
        tf.compat.v1.global_variables_initializer()

        input_var, label, paths = sample
        first_frame_path = paths[0][0].numpy().decode('utf-8')
        test_clip_id = get_video_id_from_frame_path(first_frame_path)

        clip_index = int(test_clip_id.split('_')[-1])
        video_id = str(gt_human_clips_df.loc[gt_human_clips_df['ind'] == clip_index]['video_id'].values[0])
        cps_score = gt_human_clips_df.loc[gt_human_clips_df['ind'] == clip_index]['pain'].values[0]

        print('Test clip {}, video ID {}, with CPS score {}'.format(test_clip_id, video_id, cps_score))

        input_var = tf.cast(input_var, tf.float32)
        print('\n Input var shape: {}, label shape: {}'.format(
            input_var.shape, label.shape))
        preds, merged_output = model(input_var)
        print('preds: ', preds)
        print('ground truth: ', label)
        print('preds[:, 0] shape', preds[:, 0].shape)
        guessed_score = np.max(preds, axis=1)

        print('np.max preds (confidence for guessed class)', guessed_score)

        true_class = np.argmax(label)
        true_class_score = preds[:, true_class]
        print('preds before save', preds)
        save_path = os.path.join(
            results_folder,
            test_clip_id + '_' + str(true_class) + 'g_' +
            str(np.argmax(preds)) +
            '_cs%5.4f' % true_class_score +
            'gs%5.4f' % guessed_score,
            'combined')

        print(save_path)

        result_list = [test_clip_id, video_id, cps_score, true_class, np.argmax(preds), guessed_score[0]]
        results_list.append(result_list)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    results_df = pd.DataFrame(results_list, columns=col_headers)
    results_df.to_csv(results_folder + 'results_summary.csv')


if __name__ == '__main__':
    arg_parser = arg_parser.ArgParser(len(sys.argv))
    args = arg_parser.parse()
    config_dict_module = helpers.load_module(args.config_file)
    config_dict = config_dict_module.config_dict
    config_dict['job_identifier'] = args.job_identifier
    wandb.init(project='pfr-interpretability', config=config_dict)

    all_subjects_df = pd.read_csv(args.subjects_overview)

    data_df = pd.read_csv(config_dict['data_df_path'])

    dataset, nb_steps_assuming_bs1 = make_df_for_testclips.get_dataset_from_df(
        df=data_df,
        data_columns=['pain'],
        config_dict=config_dict,
        all_subjects_df=all_subjects_df)

    run()

