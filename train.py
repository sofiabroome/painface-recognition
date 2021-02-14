from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, RMSprop
from tensorflow.keras.optimizers import SGD
from tf_slice_assign import slice_assign
import tensorflow as tf
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import models
import wandb
import time
import os

import test_and_eval

from wandb.keras import WandbCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
one = None


def train(model_instance, config_dict, train_steps, val_steps,
          train_dataset=None, val_dataset=None):
    """
    Train the model.
    :param model_instance: Model object from my file models.py | The model instance.
                           model_instance.model is the keras Sequential()-object.
    :param config_dict: {}
    :param train_steps: int
    :param val_steps: int
    :param train_dataset: tf.data.Dataset
    :param val_dataset: tf.data.Dataset
    :return: keras.Sequential() object | The (trained) model instance
    """
    print(model_instance.model.summary())
    best_model_path = create_best_model_path(config_dict)
    print('Saving best epoch model to: ', best_model_path)

    print("Setting the optimizer to {}.".format(config_dict['optimizer']))
    if config_dict['optimizer'] == 'adam':
        optimizer = Adam(lr=config_dict['lr'])
    if config_dict['optimizer'] == 'adadelta':
        optimizer = Adadelta(lr=config_dict['lr'])
    if config_dict['optimizer'] == 'adagrad':
        optimizer = Adagrad(lr=config_dict['lr'])

    print("Using binary crossentropy and binary accuracy metrics.")
    if config_dict['fine_tune']:
        model_instance.model.load_weights(config_dict['checkpoint']).expect_partial()
    else:
        model_instance.model.compile(loss='binary_crossentropy',
                                     optimizer=optimizer,
                                     metrics=['binary_accuracy'])

    if config_dict['train_mode'] == 'keras':
        keras_train(model_instance.model, best_model_path,
                    config_dict, train_steps, val_steps,
                    train_dataset, val_dataset)

    if config_dict['train_mode'] == 'low_level':
        last_model_path = create_last_model_path(config_dict)
        print('Saving last epoch model to: ', last_model_path)
        low_level_train(model_instance.model, best_model_path, last_model_path,
                        optimizer, config_dict, train_steps,
                        val_steps, train_dataset, val_dataset)

    return best_model_path


def keras_train(model, ckpt_path, config_dict, train_steps, val_steps,
                train_dataset, val_dataset):
    early_stopping = EarlyStopping(monitor=config_dict['monitor'],
                                   patience=config_dict['early_stopping'])
    checkpointer = ModelCheckpoint(filepath=ckpt_path,
                                   monitor=config_dict['monitor'],
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=True,
                                   mode=config_dict['monitor_mode'])

    binacc_train_history = BinAccTrainHistory()
    binacc_test_history = BinAccTestHistory()

    print('TRAIN STEPS: ', train_steps)
    print('VAL STEPS: ', val_steps)

    if config_dict['model'] == 'inception_4d_input':
        for layer in model.layers[:249]:
            layer.trainable = False
        for layer in model.layers[249:]:
            layer.trainable = True

        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                      loss='binary_crossentropy')

    model.fit(x=train_dataset,
              steps_per_epoch=train_steps,
              epochs=config_dict['nb_epochs'],
              callbacks=[early_stopping, checkpointer,
                         binacc_test_history, binacc_train_history,
                         WandbCallback(monitor=config_dict['monitor'])],
              validation_data=val_dataset,
              validation_steps=val_steps,
              verbose=1,
              workers=config_dict['nb_workers'])

    plot_training(binacc_test_history, binacc_train_history, config_dict)


def get_k_max_scores_per_class(preds_batch, lengths_batch, batch_size, config_dict):
    """
    :param preds_batch: tf.Tensor, shape=(batch_size, video_pad_length, nb_labels), dtype=float32
    :param lengths_batch: tf.Tensor, shape=(batch_size,) dtype=int32
    :param batch_size: int
    :param config_dict: {}
    :return: tf.Tensor: shape=(batch_size, nb_labels), dtype=float32
    """
    k_batch = tf.cast(
        tf.math.ceil(config_dict['k_mil_fraction'] * tf.cast(lengths_batch, dtype=tf.float32)),
        dtype=tf.int32)
    kmax_scores = []
    # Need to loop over samples because tf.math.top_k requires a scalar.
    for sample_index in range(batch_size):
        preds_sample = preds_batch[sample_index, :]
        preds_sample = tf.transpose(preds_sample)  # tf.math.top_k operates on rows.
        k_preds, inds = tf.math.top_k(preds_sample, k_batch[sample_index])
        avg_kmax_scores = tf.cast(1 / k_batch[sample_index], dtype=tf.float32) * tf.reduce_sum(k_preds, axis=1)
        kmax_scores.append(avg_kmax_scores)

    kmax_scores = tf.convert_to_tensor(kmax_scores)
    return kmax_scores


def get_mil_loss(kmax_scores, y_batch):
    """
    :param kmax_scores: tf.Tensor, dtype=float32, shape [batch_size, nb_labels]
    :param y_batch: tf.Tensor, dtype=float32, shape [batch_size, nb_labels]
    :return: tf.float32
    """
    kmax_distribution = tf.keras.layers.Activation('softmax')(kmax_scores)
    label_indices = tf.argmax(y_batch, axis=1)
    mils = tf.gather(kmax_distribution, label_indices, batch_dims=1)
    mils_log = tf.math.log(mils)
    mils_sum = tf.reduce_sum(mils_log)
    return mils_sum


def batch_calc_TV_norm(batch_preds, lengths_batch, p=3, q=3):
    """"
    Calculates the Total Variational Norm by summing the differences of the values
    in between the different positions of the sequences.
    """

    diff = batch_preds[:, 1:] - batch_preds[:, :-1]
    tot_var = tf.reduce_sum(tf.abs(diff)**p, axis=1)
    tot_var = tot_var ** (1/p)
    tot_var = tot_var ** q
    # Divide by sequence length
    tot_var /= tf.cast(lengths_batch, dtype=tf.float32)

    return tot_var


def get_sparse_pain_loss(y_batch, preds_batch, lengths_batch, config_dict):
    batch_size = y_batch.shape[0]  # last batch may be smaller

    if len(y_batch.shape) == 3:
        y_batch = y_batch[:, 0, :]  # Take first (video-level label)

    kmax_scores = get_k_max_scores_per_class(preds_batch, lengths_batch, batch_size, config_dict)
    mil = get_mil_loss(kmax_scores, y_batch)
    batch_indicator_nopain = tf.cast(y_batch[:, 0], dtype=tf.float32)
    batch_indicator_pain = tf.cast(y_batch[:, 1], dtype=tf.float32)

    pain_tv = batch_calc_TV_norm(preds_batch[:, :, 1], lengths_batch)

    if config_dict['tv_weight_nopain'] == 0:
        tv_nopain = 0
    else:
        tv_nopain = config_dict['tv_weight_nopain'] * tf.reduce_sum(
            batch_indicator_nopain * pain_tv)
    if config_dict['tv_weight_pain'] == 0:
        tv_pain = 0
    else:
        tv_pain = config_dict['tv_weight_pain'] * tf.reduce_sum(
            batch_indicator_pain * pain_tv)
    total_loss = tv_nopain + tv_pain - mil
    if config_dict['l1_nopain']:
        l1_batch_vector = batch_indicator_nopain * tf.reduce_sum(preds_batch[:, :, 1], axis=1)
        l1_nopain_scalar = tf.reduce_sum(l1_batch_vector)
        total_loss += l1_nopain_scalar

    return total_loss, tv_pain, tv_nopain, mil


def mask_out_padding_predictions(preds_batch, lengths_batch, batch_size, pad_length):
    zeros = tf.zeros_like(preds_batch)

    mask_tensor = tf.TensorArray(tf.float32, size=batch_size)

    global one
    if one is None:  # Cannot recreate tf.Variables in each step.
        one = tf.Variable(tf.ones([pad_length, 1]))

    for u in range(batch_size):
        one.assign(tf.ones([pad_length, 1]))

        # Put zeros in the mask past the (variable) sequence length
        indices = tf.where([(i >= lengths_batch[u]) for i in range(pad_length)])
        zeros_for_mask = tf.gather(zeros[0, :, 0], indices)
        mask = tf.compat.v1.scatter_nd_update(one, indices, zeros_for_mask)

        # Need mask for both classes: (pad_length x nb_labels)
        masks = tf.stack([mask, mask], axis=1)
        mask_tensor = mask_tensor.write(u, masks)

    mask_tensor = mask_tensor.stack()
    mask_tensor = tf.reshape(mask_tensor, preds_batch.shape)
    preds_batch = tf.keras.layers.multiply([preds_batch, mask_tensor])
    return preds_batch


def get_nb_clips_per_video(batch_video_id, df):
    nb_in_batch = batch_video_id.shape[0]
    nb_clips_batch = []
    for video_ind in range(nb_in_batch):
        video_id = batch_video_id[video_ind].numpy().decode("utf-8")
        nb_clips = df.loc[df['video_id'] == video_id]['length'].values
        nb_clips_batch.append(nb_clips[0])
    return tf.convert_to_tensor(nb_clips_batch, dtype=tf.int32)


def video_level_train(model, config_dict, train_dataset, val_dataset=None):
    """
    Train a simple model on features on video-level, since we have sparse
    pain behavior in the LPS data.
    """
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_old = -1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config_dict['lr'],
        decay_steps=40,
        decay_rate=0.96,
        staircase=True)
    # optimizer = Adam(learning_rate=lr_schedule)
    optimizer = RMSprop(learning_rate=lr_schedule)
    last_ckpt_path = create_last_model_path(config_dict)
    best_ckpt_path = create_last_model_path(config_dict)

    df_video_lengths = pd.read_csv('metadata/video_lengths.csv')

    train_steps = len([sample for sample in train_dataset])
    if not config_dict['val_mode'] == 'no_val':
        val_steps = len([sample for sample in val_dataset])
    epochs_not_improved = 0

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[config_dict['video_batch_size_train'], config_dict['video_pad_length'],
            config_dict['feature_dim']], dtype=tf.float32),
        tf.TensorSpec(shape=[config_dict['video_batch_size_train'], config_dict['video_pad_length'], 2], dtype=tf.float32),
        tf.TensorSpec(shape=[config_dict['video_batch_size_train'], config_dict['video_pad_length'], 2], dtype=tf.int32),
        tf.TensorSpec(shape=[config_dict['video_batch_size_train'],], dtype=tf.int32)))
    def train_step(x, preds, y, lengths):
        with tf.GradientTape() as tape:
            if config_dict['video_loss'] == 'cross_entropy':
                preds = model([x, preds], training=True)
                y = y[:, 0, :]
                loss = loss_fn(y, preds)
            if config_dict['video_loss'] == 'mil':
                preds_seqs = []
                for i in range(config_dict['mc_dropout_samples']):
                    preds_seq = model([x, preds], training=True)
                    preds_seq = mask_out_padding_predictions(preds_seq, lengths, config_dict['video_batch_size_train'], config_dict['video_pad_length'])
                    preds_seqs.append(preds_seq)
                preds_seq = tf.math.reduce_mean(preds_seqs, axis=0)
                sparse_loss, tv_p, tv_np, mil = get_sparse_pain_loss(y, preds_seq, lengths, config_dict)
                loss = sparse_loss
                preds_mil = test_and_eval.evaluate_sparse_pain(preds_seq, lengths, config_dict)
                preds = preds_mil
                y = y[:, 0, :]
            if config_dict['video_loss'] == 'mil_ce':
                preds_seq, preds_one = model([x, preds], training=True)
                y_one = y[:, 0, :]
                ce_loss = loss_fn(y_one, preds_one)
                sparse_loss, tv_p, tv_np, mil  = get_sparse_pain_loss(y, preds_seq, lengths, config_dict)
                preds_mil = test_and_eval.evaluate_sparse_pain(preds_seq, lengths, config_dict)
                preds = preds_mil
                y = y[:, 0, :]
                loss = ce_loss + sparse_loss
            l2_loss = config_dict['l2_weight'] * tf.reduce_sum([tf.nn.l2_loss(x) for x in model.trainable_weights if 'bias' not in x.name])
            total_loss = loss + l2_loss
        grads = tape.gradient(total_loss, model.trainable_weights)
        grads_names = [tw.name for tw in model.trainable_weights]
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y, preds)
        if 'mil' in config_dict['video_loss']:
            return grads, model.trainable_weights, grads_names, total_loss, l2_loss, tv_p, tv_np, mil
        else:
            return grads, model.trainable_weights, grads_names, total_loss, l2_loss

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[config_dict['video_batch_size_test'], config_dict['video_pad_length'],
            config_dict['feature_dim']], dtype=tf.float32),
        tf.TensorSpec(shape=[config_dict['video_batch_size_test'], config_dict['video_pad_length'], 2], dtype=tf.float32),
        tf.TensorSpec(shape=[config_dict['video_batch_size_test'], config_dict['video_pad_length'], 2], dtype=tf.int32),
        tf.TensorSpec(shape=[config_dict['video_batch_size_test'],], dtype=tf.int32)))
    def validation_step(x, preds, y, lengths):
        if config_dict['video_loss'] == 'cross_entropy':
            preds = model([x, preds], training=False)
            y = y[:, 0, :]
            loss = loss_fn(y, preds)
        if config_dict['video_loss'] == 'mil':
            preds_seqs = []
            for i in range(config_dict['mc_dropout_samples']):
                training=False if config_dict['mc_dropout_samples'] == 1 else True
                preds_seq = model([x, preds], training=training)
                preds_seq = mask_out_padding_predictions(preds_seq, lengths, config_dict['video_batch_size_test'], config_dict['video_pad_length'])
                preds_seqs.append(preds_seq)
            preds_seq = tf.math.reduce_mean(preds_seqs, axis=0)
            sparse_loss, tv_p, tv_np, mil = get_sparse_pain_loss(y, preds_seq, lengths, config_dict)
            loss = sparse_loss
            preds_mil = test_and_eval.evaluate_sparse_pain(preds_seq, lengths, config_dict)
            preds = preds_mil
            y = y[:, 0, :]
        if config_dict['video_loss'] == 'mil_ce':
            preds_seq, preds_one = model([x, preds], training=False)
            preds_seq = mask_out_padding_predictions(preds_seq, lengths, config_dict['video_batch_size_test'], config_dict['video_pad_length'])
            preds_one = tf.keras.layers.Activation('softmax')(preds_one)
            y_one = y[:, 0, :]
            ce_loss = loss_fn(y_one, preds_one)
            sparse_loss, tv_p, tv_np, mil  = get_sparse_pain_loss(y, preds_seq, lengths, config_dict)
            loss = ce_loss + sparse_loss
            preds_mil = test_and_eval.evaluate_sparse_pain(preds_seq, lengths, config_dict)
            preds = 1/2 * (preds_one + preds_mil)
            y = y[:, 0, :]
        val_acc_metric.update_state(y, preds)
        if 'mil' in config_dict['video_loss']:
            return loss, tv_p, tv_np, mil
        else:
            return loss

    k_mil_drop = config_dict['k_mil_fraction_start'] - config_dict['k_mil_fraction_end']
    nb_decrements = k_mil_drop/config_dict['k_mil_fraction_decrement_step']
    k_mil_decreasing_nb_decrements_counter = 0
    k_mil_decreasing_epoch_counter = 0
    config_dict['k_mil_fraction'] = config_dict['k_mil_fraction_start']

    for epoch in range(config_dict['video_nb_epochs']):
        print('\nStart of epoch %d' % (epoch,))
        wandb.log({'epoch': epoch})

        if (k_mil_decreasing_nb_decrements_counter < nb_decrements) and \
            (k_mil_decreasing_epoch_counter == config_dict['k_mil_fraction_nb_epochs_to_decrease']):
            config_dict['k_mil_fraction'] -= config_dict['k_mil_fraction_decrement_step']
            k_mil_decreasing_epoch_counter = 0
            k_mil_decreasing_nb_decrements_counter += 1

        k_mil_decreasing_epoch_counter += 1

        print('\nk-MIL fraction after every-epoch update: {}'.format(config_dict['k_mil_fraction']))
        
        start_time = time.time()

        with tqdm(total=train_steps) as pbar:
            for step, sample in enumerate(train_dataset):
                # step_start_time = time.time()
                pbar.update(1)
                feats_batch, preds_batch, labels_batch, video_id = sample
                
                lengths_batch = get_nb_clips_per_video(video_id, df_video_lengths)

                # print('\n Video ID: ', video_id)

                if 'mil' in config_dict['video_loss']:
                    grads, trainable_weights, grads_names, loss_value, l2_loss, tv_p, tv_np, mil = train_step(
                        feats_batch, preds_batch, labels_batch, lengths_batch)
                    wandb.log({'tv_nopain': tv_np.numpy()})
                    wandb.log({'tv_pain': tv_p.numpy()})
                    wandb.log({'mil': mil.numpy()})
                    print(optimizer._decayed_lr('float32').numpy())
                    wandb.log({'lr': optimizer._decayed_lr('float32').numpy()})
                else:
                    grads, trainable_weights, grads_names, loss_value, l2_loss = train_step(
                        feats_batch, preds_batch, labels_batch, lengths_batch)

                # step_time = time.time() - step_start_time
                # print('Step time: %.2f' % step_time)
                wandb.log({'train_loss': loss_value.numpy()})
                wandb.log({'l2_loss': l2_loss.numpy()})
                if step % config_dict['print_loss_every'] == 0:
                    for ind, g in enumerate(grads):
                        wandb.log({'grad_' + grads_names[ind].numpy().decode("utf-8"): wandb.Histogram(g.numpy())})
                    for ind, tw in enumerate(trainable_weights):
                        wandb.log({'param_' + grads_names[ind].numpy().decode("utf-8"): wandb.Histogram(tw.numpy())})

                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value.numpy()))
                    )
                    print("Seen so far: %d samples" %
                          ((step + 1) * config_dict['batch_size']))
                    print('\n GRADS:')
                    print([grad.numpy() for grad in grads][0])
                    print('\n Video ID: ', video_id)
                    print(feats_batch.shape, preds_batch.shape, labels_batch.shape)

        train_acc = train_acc_metric.result()
        wandb.log({'train_acc': train_acc})
        print('Training acc over epoch: %.4f' % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        if not config_dict['val_mode'] == 'no_val':

            with tqdm(total=val_steps) as pbar:
                for step, sample in enumerate(val_dataset):
                    if step > val_steps:
                        break
                    pbar.update(1)
                    # step_start_time = time.time()
                    feats_batch, preds_batch, labels_batch, video_id = sample
                    lengths_batch = get_nb_clips_per_video(video_id, df_video_lengths)
                    # print('\n Video ID: ', video_id)
                    if 'mil' in config_dict['video_loss']:
                        loss_value, tv_p, tv_np, mil = validation_step(
                            feats_batch, preds_batch, labels_batch, lengths_batch)
                    else:
                        loss_value = validation_step(
                            feats_batch, preds_batch, labels_batch, lengths_batch)

                    # step_time = time.time() - step_start_time
                    # print('Step time: %.2f' % step_time)

            wandb.log({'val_loss': loss_value.numpy()})
            if 'mil' in config_dict['video_loss']:
                wandb.log({'val_tv_nopain': tv_np.numpy()})
                wandb.log({'val_tv_pain': tv_p.numpy()})
                wandb.log({'val_mil': mil.numpy()})
            val_acc = val_acc_metric.result()
            wandb.log({'val_acc': val_acc})
            print("Validation acc: %.4f" % (float(val_acc),))

            if val_acc > val_acc_old:
                print('The validation acc improved, saving checkpoint...')
                epochs_not_improved = 0
                model.save_weights(best_ckpt_path)
                val_acc_old = val_acc
            else:
                print('The validation acc did not improve, incrementing the early-stopping counter...')
                epochs_not_improved += 1
                if epochs_not_improved == config_dict['video_early_stopping']:
                    break

            val_acc_metric.reset_states()

        print('\n Saving checkpoint to {} after epoch {}'.format(last_ckpt_path, epoch))
        model.save_weights(last_ckpt_path)
        print("Epoch time taken: %.2fs" % (time.time() - start_time))
    return best_ckpt_path


def low_level_distributed_train(model, ckpt_path, last_ckpt_path, optimizer,
                    config_dict, train_steps, val_steps=None,
                    train_dataset=None, val_dataset=None):
    """
    Train a model in "low-level" tf with standard, dense supervision.
    """
    with strategy.scope():
        # Set reduction to `none` so we can do the reduction afterwards and divide by
        # global batch size.
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_old = -1
    mirrored_strategy = tf.distribute.MirroredStrategy()
    print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dataset = strategy.experimental_distribute_dataset(val_dataset)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss = loss_fn(y, preds)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y, preds)
        return loss

    @tf.function
    def validation_step(x, y):
        preds = model(x, training=False)
        loss = loss_fn(y, preds)
        val_acc_metric.update_state(y, preds)
        return loss
    
    epochs_not_improved = 0
    print('\n Saving checkpoint to {} before first epoch'.format(last_ckpt_path))
    model.save_weights(last_ckpt_path)
    for epoch in range(config_dict['nb_epochs']):
        print('\nStart of epoch %d' % (epoch,))
        wandb.log({'epoch': epoch})
        start_time = time.time()

        with tqdm(total=train_steps) as pbar:
            for step, sample in enumerate(train_dataset):
                if step > train_steps:
                    break
                pbar.update(1)
                # step_start_time = time.time()
                x_batch_train, y_batch_train = sample
                # print(x_batch_train.shape, y_batch_train.shape)
                # x_batch_train, y_batch_train, paths = sample
                loss_value = train_step(x_batch_train, y_batch_train)
                # step_time = time.time() - step_start_time
                # print('Step time: %.2f' % step_time)
                wandb.log({'train_loss': loss_value.numpy()})

                if step % config_dict['print_loss_every'] == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value.numpy()))
                    )
                    print("Seen so far: %d samples" %
                          ((step + 1) * config_dict['batch_size']))

        train_acc = train_acc_metric.result()
        wandb.log({'train_acc': train_acc})
        print('Training acc over epoch: %.4f' % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        if not config_dict['val_mode'] == 'no_val':

            with tqdm(total=val_steps) as pbar:
                for step, sample in enumerate(val_dataset):
                    if step > val_steps:
                        break
                    pbar.update(1)
                    # step_start_time = time.time()
                    x_batch_val, y_batch_val = sample
                    # x_batch_val, y_batch_val, paths = sample
                    loss_value = validation_step(x_batch_val, y_batch_val)
                    # step_time = time.time() - step_start_time
                    # print('Step time: %.2f' % step_time)

            wandb.log({'val_loss': loss_value.numpy()})
            val_acc = val_acc_metric.result()
            wandb.log({'val_acc': val_acc})
            print("Validation acc: %.4f" % (float(val_acc),))

            if val_acc > val_acc_old:
                print('The validation acc improved, saving checkpoint...')
                model.save_weights(ckpt_path)
                print('Resetting epochs not improved.')
                epochs_not_improved = 0
                val_acc_old = val_acc
            else:
                epochs_not_improved += 1
                if epochs_not_improved == config_dict['early_stopping']:
                    break
                    
            val_acc_metric.reset_states()
        print('\n Saving checkpoint to {} after epoch {}'.format(last_ckpt_path, epoch))
        model.save_weights(last_ckpt_path)
        print("Epoch time taken: %.2fs" % (time.time() - start_time))


def low_level_train(model, ckpt_path, last_ckpt_path, optimizer,
                    config_dict, train_steps, val_steps=None,
                    train_dataset=None, val_dataset=None):
    """
    Train a model in "low-level" tf with standard, dense supervision.
    """

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_old = -1

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss = loss_fn(y, preds)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y, preds)
        return loss

    @tf.function
    def validation_step(x, y):
        preds = model(x, training=False)
        loss = loss_fn(y, preds)
        val_acc_metric.update_state(y, preds)
        return loss
    
    epochs_not_improved = 0
    print('\n Saving checkpoint to {} before first epoch'.format(last_ckpt_path))
    model.save_weights(last_ckpt_path)
    for epoch in range(config_dict['nb_epochs']):
        print('\nStart of epoch %d' % (epoch,))
        wandb.log({'epoch': epoch})
        start_time = time.time()

        with tqdm(total=train_steps) as pbar:
            for step, sample in enumerate(train_dataset):
                if step > train_steps:
                    break
                pbar.update(1)
                # step_start_time = time.time()
                x_batch_train, y_batch_train = sample
                # print(x_batch_train.shape, y_batch_train.shape)
                # x_batch_train, y_batch_train, paths = sample
                loss_value = train_step(x_batch_train, y_batch_train)
                # step_time = time.time() - step_start_time
                # print('Step time: %.2f' % step_time)
                wandb.log({'train_loss': loss_value.numpy()})

                if step % config_dict['print_loss_every'] == 0:
                    # for i in range(5):
                    #     wandb.log({"step_{}_frame_{}".format(step, i): [wandb.Image(x_batch_train[0,0,i,:], caption="frame")]})
                    #     wandb.log({"step_{}_flow_{}".format(step, i): [wandb.Image(x_batch_train[0,1,i,:], caption="flow")]})
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value.numpy()))
                    )
                    print("Seen so far: %d samples" %
                          ((step + 1) * config_dict['batch_size']))

        train_acc = train_acc_metric.result()
        wandb.log({'train_acc': train_acc})
        print('Training acc over epoch: %.4f' % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        if not config_dict['val_mode'] == 'no_val':

            with tqdm(total=val_steps) as pbar:
                for step, sample in enumerate(val_dataset):
                    if step > val_steps:
                        break
                    pbar.update(1)
                    # step_start_time = time.time()
                    x_batch_val, y_batch_val = sample
                    # x_batch_val, y_batch_val, paths = sample
                    loss_value = validation_step(x_batch_val, y_batch_val)
                    # step_time = time.time() - step_start_time
                    # print('Step time: %.2f' % step_time)

            wandb.log({'val_loss': loss_value.numpy()})
            val_acc = val_acc_metric.result()
            wandb.log({'val_acc': val_acc})
            print("Validation acc: %.4f" % (float(val_acc),))

            if val_acc > val_acc_old:
                print('The validation acc improved, saving checkpoint...')
                model.save_weights(ckpt_path)
                print('Resetting epochs not improved.')
                epochs_not_improved = 0
                val_acc_old = val_acc
            else:
                epochs_not_improved += 1
                if epochs_not_improved == config_dict['early_stopping']:
                    break
                    
            val_acc_metric.reset_states()
        print('\n Saving checkpoint to {} after epoch {}'.format(last_ckpt_path, epoch))
        model.save_weights(last_ckpt_path)
        print("Epoch time taken: %.2fs" % (time.time() - start_time))


def save_features(model, config_dict, steps, dataset):
    """
    Save features to file.
    """
    if config_dict['inference_only']:
        model.load_weights(config_dict['checkpoint']).expect_partial()

    @tf.function
    def get_features_step(x):
        predictions, features = model(x, training=False)
        # # Downsample further with one MP layer, strides and kernel 2x2
        # # The result per frame is 4x4x32, if 128x128. 7x7x32 if 224x224.
        features = tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPool2D())(features)
        # # The result per frame is 1x32.
        # features = tf.keras.layers.TimeDistributed(
        #     tf.keras.layers.GlobalAveragePooling2D())(features)
        features = tf.keras.layers.Flatten()(features)
        return predictions, features

    features_to_save = []

    with tqdm(total=steps) as pbar:
        for step, sample in enumerate(dataset):
            if step > steps:
                break
            pbar.update(1)
            to_save_dict = {}
            x_batch_train, y_batch_train, paths = sample
            preds, flow_rgb_map_merge = get_features_step(x_batch_train)
            to_save_dict['paths'] = paths
            to_save_dict['preds'] = preds
            to_save_dict['features'] = flow_rgb_map_merge
            to_save_dict['y'] = y_batch_train
            features_to_save.append(to_save_dict)
    features_to_save = np.asarray(features_to_save)
    np.savez_compressed(config_dict['checkpoint'][:30] + config_dict['save_clip_feats_id'], features_to_save)


def val_split(X_train, y_train, val_fraction, batch_size, round_to_batch=True):
    if round_to_batch:
        ns = X_train.shape[0]
        num_val = int(val_fraction * ns - val_fraction * ns % batch_size)

        X_val = X_train[-num_val:, :]
        y_val = y_train[-num_val:]

        X_train = X_train[:-num_val, :]
        y_train = y_train[:-num_val:]

    return X_train, y_train, X_val, y_val


def round_to_batch_size(data_array, batch_size):
    num_rows = data_array.shape[0]
    surplus = num_rows % batch_size
    data_array_rounded = data_array[:num_rows-surplus]
    return data_array_rounded


def create_best_model_path(config_dict):
    model_path = '_'.join(('models/' + config_dict['job_identifier'],
                           'best_model', config_dict['model'])) + '.ckpt'
    return model_path


def create_last_model_path(config_dict):
    model_path = '_'.join(('models/' + config_dict['job_identifier'],
                           'last_model', config_dict['model'])) + '.ckpt'
    return model_path


class CatAccTestHistory(Callback):
    def on_train_begin(self, logs={}):
        self.cataccs = []

    def on_epoch_end(self, epoch, logs={}):
        self.cataccs.append(logs.get('val_categorical_accuracy'))


class CatAccTrainHistory(Callback):
    def on_train_begin(self, logs={}):
        self.cataccs = []

    def on_epoch_end(self, epoch, logs={}):
        self.cataccs.append(logs.get('categorical_accuracy'))


class BinAccTestHistory(Callback):
    def on_train_begin(self, logs={}):
        self.binaccs = []

    def on_epoch_end(self, epoch, logs={}):
        self.binaccs.append(logs.get('val_binary_accuracy'))


class BinAccTrainHistory(Callback):
    def on_train_begin(self, logs={}):
        self.binaccs = []

    def on_epoch_end(self, epoch, logs={}):
        self.binaccs.append(logs.get('binary_accuracy'))


def plot_training(test_history,
                  train_history,
                  config_dict):
    plt.plot(test_history.binaccs, label='Validation accuracy')
    plt.plot(train_history.binaccs, label='Training accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('results/train_plot_' + config_dict['model'] + '_' +
                config_dict['job_identifier'] + '.png')
    plt.close()

