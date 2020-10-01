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


def get_variables_to_restore():
    variables_to_restore = {}
    for variable in tf.global_variables():
        if variable.name.startswith('mask'):
            continue
        elif variable.name.startswith('original_input'):
            continue
        else:
            # Variables need to be renamed to match with the checkpoint.
            variables_to_restore[variable.name.replace(':0', '')] = variable

    return variables_to_restore


def run(verbose=True, do_gradcam=True):

    if not os.path.exists(config_dict['output_folder']):
        os.makedirs(config_dict['output_folder'])

    model = models.MyModel(config_dict=config_dict).model
    model.load_weights(config_dict['checkpoint'])

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
            after_softmax = mask.perturbSequence(x, mask_var)
            if config_dict['focus_type'] == 'correct':
                label_index = tf.reshape(
                    tf.argmax(y, axis=1), [])
            if config_dict['focus_type'] == 'guessed':
                label_index = tf.reshape(
                    tf.argmax(after_softmax, axis=1), [])
            class_loss = after_softmax[:, label_index]
            # Cast as same type as l1 and TV.
            class_loss = tf.cast(class_loss, tf.float32)
            l1 = config_dict['lambda_1'] * tf.reduce_sum(tf.abs(mask_clip))
            tv = config_dict['lambda_2'] * mask.calc_TV_norm(mask_clip, p=3, q=3)
            loss = l1 + tv + class_loss
        grads = tape.gradient(loss, time_mask.trainable_weights)
        optimizer.apply_gradients(zip(grads, mask_var.trainable_weights))
        return [loss, l1, tv, class_loss]

    for ind, sample in enumerate(dataset):

        video_id = 'clip_' + str(ind)

        input_var, label = sample
        input_var = tf.cast(input_var, tf.float32)
        print('\n Input var shape: {}, label shape: {}'.format(
            input_var.shape, label.shape))
        preds = model(input_var)
        print('preds[:, 0] shape', preds[:, 0].shape)
        after_softmax_value = np.max(preds[:, 0], axis=1)
        # after_softmax_value = apply_mask_to_sequence(
        #     x=input_var,
        #     mask=np.zeros((config_dict['seq_length'])))

        print('np.argmax preds', after_softmax_value)

        # eta is for breaking out of the grad desc early if it hasn't improved
        eta = 0.00001

        time_mask = mask.init_mask(input_var,
                                   label,
                                   model,
                                   forward_fn=apply_mask_to_sequence,
                                   thresh=0.9,
                                   mode="central")
        mask_var.assign(time_mask)

        old_loss = 999999
        for nidx in range(config_dict['nb_iterations_graddescent']):

            if (nidx % 10) == 0:
                print("on nidx: ", nidx)
                print("mask_clipped is: ", mask_clip)

            losses = train_step(input_var, label)
            loss_value, l1value, tvvalue, classlossvalue = losses

            print("Total loss: {}, L1 loss: {}, TV: {}, class score: {}".format(
                loss_value, l1value, tvvalue, classlossvalue))

            if abs(old_loss - loss_value) < eta:
                break

        mask_clip = tf.nn.sigmoid(time_mask)
        save_path = os.path.join("cam_saved_images",
                                 config_dict['output_folder'],
                                 str(np.argmax(label)),
                                 video_id + 'g_' +
                                 str(np.argmax(after_softmax_value.numpy())) +
                                 '_cs%5.4f' % after_softmax_value.numpy()[:, np.argmax(label)] +
                                 'gs%5.4f' % after_softmax_value.numpy()[:, np.argmax(after_softmax_value.numpy())],
                                 'combined')

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        f = open(save_path + '/class_score_freeze_case' + video_id + '.txt', 'w+')
        f.write(str(classlossvalue))
        f.close()

        if config_dict['temporal_mask_type'] == 'reverse':
            perturbed_sequence = mask.perturb_sequence(
                input_var, time_mask, perb_type='reverse')

            after_softmax_rev = apply_mask_to_sequence(perturbed_sequence, mask_var)
            class_loss_rev = after_softmax_rev.numpy()[:, np.argmax(label)]

            f = open(save_path + '/class_score_reverse_case' + video_id + '.txt', 'w+')
            f.write(str(class_loss_rev))
            f.close()

            if verbose:
                print("Resulting mask: ", mask_clip)

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

