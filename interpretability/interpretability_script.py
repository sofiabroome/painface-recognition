import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb

import arg_parser
import helpers
import models
from interpretability import mask, gradcam as gc, interpretability_viz as viz


def build_graph():
    # First we need to recreate the same variables as in the model.
    tf.reset_default_graph()
    seq_shape = (config_dict['batch_size'],
                 config_dict['seq_length'],
                 config_dict['image_size'],
                 config_dict['image_size'], 3)

    # Build graph
    graph = tf.Graph()

    # Graph for perturb_sequence(seq, mask, perb_type) method
    # Create variable to save original input sequence
    with tf.variable_scope('original_input'):
        original_input_plhdr = tf.placeholder(tf.float32, seq_shape)
        original_input_var = tf.get_variable('original_input',
                                             seq_shape,
                                             dtype=tf.float32,
                                             trainable=False)
        original_input_assign = original_input_var.assign(original_input_plhdr)

    with tf.variable_scope('mask'):
        # Create variable for the temporal mask
        mask_plhdr = tf.placeholder(tf.float32, [config_dict['seq_length']])
        mask_var = tf.get_variable('input_mask',
                                   [config_dict['seq_length']],
                                   dtype=tf.float32,
                                   trainable=True)
        mask_assign = tf.assign(mask_var, mask_plhdr)
        mask_clip = tf.nn.sigmoid(mask_var)

    with tf.variable_scope('perturb'):

        frame_inds = tf.placeholder(tf.int32, shape=(None,), name='frame_inds')

        def recurrence(last_value, current_elem):
            update_tensor = (1 - mask_clip[current_elem]) * original_input_var[:, current_elem, :, :, :] + \
                            mask_clip[current_elem] * last_value
            return update_tensor

        perturb_op = tf.scan(fn=recurrence,
                             elems=frame_inds,
                             initializer=original_input_var[:, 0, :, :, :])
        perturb_op = tf.reshape(perturb_op, seq_shape)

    y = tf.placeholder(tf.float32, [config_dict['batch_size'], config_dict['nb_labels']])
    logits, clstm_3 = model(perturb_op)
    after_softmax = tf.nn.softmax(logits)

    return original_input_assign, mask_assign, mask_var,\
        mask_clip, y, logits, after_softmax


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


def run(dataset, verbose=True, do_gradcam=True):

    if not os.path.exists(config_dict['output_folder']):
        os.makedirs(config_dict['output_folder'])

    model = models.MyModel(config_dict=config_dict)
    model.load_weights(config_dict['checkpoint'])

    optimizer = tf.train.AdamOptimizer(
        learning_rate=config_dict['learning_rate_start'])

    # The mask is what we optimize over
    mask_var = tf.Variable(tf.ones(shape=(config_dict['seq_length'])),
                           name='mask',
                           trainable=True,
                           dtype=tf.float32)
    frame_inds = range(config_dict['seq_length'])

    @tf.function
    def forward(x):

        def recurrence(last_value, current_elem):
            update_tensor = (1 - mask_clip[current_elem]) * x[:, current_elem, :, :, :] + \
                            mask_clip[current_elem] * last_value
            return update_tensor

        masked_input = tf.scan(
            fn=recurrence, elems=frame_inds,
            initializer=x[:, 0, :, :, :])

        logits, last_clstm_output = model(masked_input)
        return tf.nn.softmax(logits)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            after_softmax = forward(x, mask_var)
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
        optimizer.apply_gradients(zip(grads, time_mask.trainable_weights))
        return [loss, l1, tv, class_loss]

    for input_var, label, video_id in dataset:

        print("Found clip of interest ", video_id)

        after_softmax_value = forward(
            mask_var=np.zeros((config_dict['seq_length'])),
            original_input=input_var,
            frame_inds=frame_inds)

        print('np.argmax preds', np.argmax(after_softmax_value.numpy()))

        # eta is for breaking out of the grad desc early if it hasn't improved
        eta = 0.00001

        time_mask = mask.init_mask(input_var,
                                   frame_inds,
                                   label, thresh=0.9,
                                   mode="central",
                                   forward_fn=forward(input_var),
                                   mask_pert_type=config_dict['temporal_mask_type'])
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

            after_softmax_rev = forward(perturbed_sequence, label, config_dict['seq_length'])
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

                # gradcam = gc.get_gradcam(logits, last_clstm_output, label,
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
                input_var,
                mask.perturb_sequence(input_var, time_mask, perb_type='reverse'),
                time_mask, root_dir=save_path, case=video_id, mark_imgs=True, iter_test=False)


if __name__ == '__main__':
    arg_parser = arg_parser.ArgParser(len(sys.argv))
    args = arg_parser.parse()
    config_dict_module = helpers.load_module(args.config_file)
    config_dict = config_dict_module.config_dict
    if config_dict['val_mode'] == 'no_val':
        assert (config_dict['train_mode'] == 'low_level'), \
            'no_val requires low level train mode'

    config_dict['job_identifier'] = args.job_identifier
    print('Job identifier: ', args.job_identifier)
    wandb.init(project='pfr', config=config_dict)

    all_subjects_df = pd.read_csv(args.subjects_overview)

    # Run the whole program, from preparing the data to evaluating
    # the model's test performance
    run()

