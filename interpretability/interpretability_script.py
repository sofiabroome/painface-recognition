import sys
sys.path.append('..')
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import arg_parser
import helpers
import models
from interpretability import mask, gradcam as gc, interpretability_viz as viz
from data_scripts import make_df_for_testclips


def run(verbose=True, do_gradcam=True):

    if not os.path.exists(config_dict['output_folder']):
        os.makedirs(config_dict['output_folder'])

    model = models.MyModel(config_dict=config_dict).model
    model.load_weights(config_dict['checkpoint']).expect_partial()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config_dict['lr'])

    # The mask is what we optimize over
    mask_var = tf.Variable(tf.ones(shape=([config_dict['seq_length']])),
                           name='mask',
                           trainable=True,
                           dtype=tf.float32)
    frame_inds = tf.range(config_dict['seq_length'], dtype=tf.int32)
    print(frame_inds)

    @tf.function
    def apply_mask_to_sequence(x, mask):

        def recurrence(last_value, current_elem):
            print('current elem', current_elem)
            update_tensor = (1 - mask[current_elem]) * x[:, current_elem, :, :, :] + \
                            mask[current_elem] * last_value
            print(update_tensor.shape)
            return update_tensor
        masked_input = tf.scan(
            fn=recurrence, elems=frame_inds,
            initializer=x[:, 0, :, :, :])

        logits, last_clstm_output = model(masked_input)
        return tf.nn.softmax(logits)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            # tape.watch(mask_var)
            # # The mask should always be "clipped" here.
            # mask_clip = tf.sigmoid(mask_var)
            # tape.watch(mask_clip)
            # print("mask_clip is: ", mask_clip)
            print('\nmask_var after sigmoid', mask_var)
            after_softmax = model(mask.perturbSequence(x, mask_var))
            print(after_softmax)
            print('\nmask_var after perturbseq', mask_var)
            if config_dict['focus_type'] == 'correct':
                label_index = tf.reshape(
                    tf.argmax(y, axis=1), [])
            if config_dict['focus_type'] == 'guessed':
                label_index = tf.reshape(
                    tf.argmax(after_softmax, axis=1), [])
            class_loss = after_softmax[:, label_index]
            # Cast as same type as l1 and TV.
            class_loss = tf.cast(class_loss, tf.float32)
            l1 = config_dict['lambda_1'] * tf.reduce_sum(
                tf.abs(mask_var))
            tv = config_dict['lambda_2'] * mask.calc_TV_norm(
                mask_var, config_dict)
            loss = l1 + tv + class_loss
        print('\nmask_var after loss comp', mask_var)
        grads = tape.gradient(loss, mask_var)
        print('\nmask_var after gradient', mask_var)
        print('grads', grads)
        optimizer.apply_gradients(zip([grads], [mask_var]))
        return [loss, l1, tv, class_loss]

    for sample_ind, sample in enumerate(dataset):

        video_id = 'clip_' + str(sample_ind)

        input_var, label = sample
        input_var = tf.cast(input_var, tf.float32)
        print('\n Input var shape: {}, label shape: {}'.format(
            input_var.shape, label.shape))
        preds = model(input_var)
        print(preds)
        print('preds[:, 0] shape', preds[:, 0].shape)
        guessed_score = np.max(preds, axis=1)

        print('np.max preds', guessed_score)

        # eta is for breaking out of the grad desc early if it hasn't improved
        eta = 0.00001

        time_mask = mask.init_mask(input_var,
                                   label,
                                   model,
                                   thresh=0.9,
                                   mode="central")
        mask_var.assign(time_mask)

        print('\nmask_var', mask_var)

        old_loss = 999999
        for step in range(config_dict['nb_iterations_graddescent']):

            if (step % 10) == 0:
                print("on step: ", step)

            losses = train_step(input_var, label)
            loss_value, l1value, tvvalue, classlossvalue = losses

            print("Total loss: {}, L1 loss: {}, TV: {}, class score: {}".format(
                loss_value, l1value, tvvalue, classlossvalue))

            if abs(old_loss - loss_value) < eta:
                break
        time_mask = tf.sigmoid(mask_var)
        true_class = np.argmax(label)
        true_class_score = preds[:, true_class]
        print('preds before save', preds)
        save_path = os.path.join('cam_saved_images',
                                 config_dict['output_folder'],
                                 str(true_class),
                                 video_id + 'g_' +
                                 str(np.argmax(preds)) +
                                 '_cs%5.4f' % true_class_score +
                                 'gs%5.4f' % guessed_score,
                                 'combined')

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        f = open(save_path + '/class_score_freeze_case' + video_id + '.txt', 'w+')
        f.write(str(classlossvalue))
        f.close()

        if config_dict['temporal_mask_type'] == 'reverse':
            perturbed_sequence = mask.perturb_sequence(
                input_var, time_mask, perb_type='reverse')

            after_softmax_rev = model(perturbed_sequence)
            class_loss_rev = after_softmax_rev[np.argmax(label)]

            f = open(save_path + '/class_score_reverse_case' + video_id + '.txt', 'w+')
            f.write(str(class_loss_rev))
            f.close()

            if verbose:
                print("Resulting mask: ", time_mask)

            if do_gradcam:

                if config_dict['focus_type'] == 'guessed':
                    target_index = np.argmax(after_softmax_rev.numpy())
                if config_dict['focus_type'] == 'correct':
                    target_index = np.argmax(label)

                # gradcam = gc.get_gradcam(config_dict, logits, last_clstm_output, label,
                #                          input_var, mask_clip, frame_inds,
                #                          input_var, label, target_index,
                #                          config_dict['image_size'],
                #                          config_dict['image_size'])

                # '''beginning of gradcam write to disk'''

                # os.makedirs(save_path, exist_ok=True)

                # viz.create_image_arrays(
                #     input_var, gradcam, time_mask,
                #     save_path, video_id, 'freeze',
                #     config_dict['image_size'], config_dict['image_size'])

                # if config_dict['temporal_mask_type'] == 'reverse':
                #     # Also create the image arrays for the reverse operation.
                #     viz.create_image_arrays(
                #         input_var, gradcam, time_mask,
                #         save_path, video_id, 'reverse',
                #         config_dict['image_size'], config_dict['image_size'])

            viz.visualize_results(
                config_dict,
                input_var,
                mask.perturb_sequence(input_var, time_mask, perb_type='reverse'),
                time_mask, root_dir=save_path, case=video_id, mark_imgs=True, iter_test=False)


if __name__ == '__main__':
    arg_parser = arg_parser.ArgParser(len(sys.argv))
    args = arg_parser.parse()
    config_dict_module = helpers.load_module(args.config_file)
    config_dict = config_dict_module.config_dict

    all_subjects_df = pd.read_csv(args.subjects_overview)

    data_df = pd.read_csv('../data/lps/random_clips_lps/'
                          'jpg_128_128_2fps/test_clip_frames.csv')

    dataset = make_df_for_testclips.get_dataset_from_df(df=data_df,
                                               data_columns=['pain'],
                                               config_dict=config_dict,
                                               all_subjects_df=all_subjects_df)

    # Run the whole program, from preparing the data to evaluating
    # the model's test performance
    run(dataset)

