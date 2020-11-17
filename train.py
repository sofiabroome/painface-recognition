from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, RMSprop
from tensorflow.keras.optimizers import SGD
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

def get_k_max_scores_per_class(y_batch, preds_batch, lengths_batch, batch_size, config_dict):
    kmax_scores = []
    for sample_index in range(batch_size):
        sample_class_kmax_scores = []
        seq_length = lengths_batch[sample_index]
        # padded_indices = [i for i in range(y_batch.shape[1]) if
        #                   (y_batch[sample_index, i, 0] == 0 and y_batch[sample_index, i, 1] == 0)]

        # if len(padded_indices) == 0:
        #     seq_length = config_dict['video_pad_length']
        # else:
        #     seq_length = padded_indices[0]
        # print(seq_length)
        # print(preds_batch.shape)
        
        k = tf.cast(tf.math.ceil(config_dict['k_mil_fraction'] * tf.cast(seq_length, dtype=tf.float32)), dtype=tf.int32)
        # print('\n', k, seq_length)
        for class_index in range(config_dict['nb_labels']):
            # Just need label index because the other would be zeroed out.
            # preds_nopad = preds_batch[sample_index, :, class_index][:seq_length]
            preds_nopad = preds_batch[sample_index, :, class_index]
            k_preds, indices = tf.math.top_k(preds_nopad, k)
            sample_class_kmax_score = tf.cast(1/k, dtype=tf.float32) * tf.reduce_sum(k_preds)
            sample_class_kmax_scores.append(sample_class_kmax_score)
        kmax_scores.append(sample_class_kmax_scores)
    kmax_scores = tf.convert_to_tensor(kmax_scores)
    return kmax_scores


def get_sparse_pain_loss(y_batch, preds_batch, lengths_batch, config_dict):
    batch_size = y_batch.shape[0]  # last batch may be smaller
    # seq_lengths = [[i for i in range(y_batch.shape[1]) if (y_batch[b, i, 0] == 0 and y_batch[b, i, 1] == 0)][0]
    #                for b in range(batch_size)]
    # print(seq_lengths)

    def get_mil_loss(kmax_scores):
        kmax_distribution = tf.keras.layers.Activation('softmax')(kmax_scores)
        mils = 0
        # for sample_index in range(config_dict['video_batch_size']):
        for sample_index in range(batch_size):
            label_index = tf.argmax(y_batch[sample_index, 0, :])  # Take first (video-level label)
            mil = tf.math.log(kmax_distribution[sample_index, label_index])
            mils += mil
        # return mils/config_dict['video_batch_size']
        # return mils/batch_size
        return mils

    kmax_scores = get_k_max_scores_per_class(y_batch, preds_batch, lengths_batch, batch_size, config_dict)
    mil = get_mil_loss(kmax_scores)
    batch_indicator_nopain = tf.cast(y_batch[:, 0, 0], dtype=tf.float32)
    batch_indicator_pain = tf.cast(y_batch[:, 0, 1], dtype=tf.float32)
    tv_nopain = config_dict['tv_weight_nopain'] * tf.reduce_sum(
        batch_indicator_nopain * batch_calc_TV_norm(preds_batch[:, :, 0],
                                                    lengths_batch))
    tv_pain = config_dict['tv_weight_pain'] * tf.reduce_sum(
        batch_indicator_pain * batch_calc_TV_norm(preds_batch[:, :, 1],
                                                  lengths_batch))
    # print('tv pain, ', tv_pain)
    # print('tv no pain, ', tv_nopain)
    # print('mil', mil)

    total_loss = tv_nopain + tv_pain - mil
    # total_loss = -mil

    return total_loss, tv_pain, tv_nopain, mil

def batch_calc_TV_norm(batch_vectors, lengths_batch, p=3, q=3):
    """"
    Calculates the Total Variational Norm by summing the differences of the values
    in between the different positions in the mask.
    p=3 and q=3 are defaults from the paper.
    """
    val = tf.cast(0, dtype=tf.float32)
    # import ipdb; ipdb.set_trace()
    batch_size = batch_vectors.shape[0]
    # vals = []
    vals = tf.TensorArray(tf.float32, size=batch_size)
    for vector_index in tf.range(batch_size):
        vector = batch_vectors[vector_index]
        vector_length = tf.cast(lengths_batch[vector_index], dtype=tf.int32)
        for u in tf.range(1, vector_length - 1):
            val += tf.abs(vector[u - 1] - vector[u]) ** p
            val += tf.abs(vector[u + 1] - vector[u]) ** p
        val = val ** (1 / p)
        val = val ** q
        # vals.append(val)
        vals = vals.write(vector_index, val)
    # return tf.convert_to_tensor(vals, dtype=tf.float32)
    return vals.stack()
    # return vals


def get_nb_clips_per_video(batch_video_id, df):
    nb_in_batch = batch_video_id.shape[0]
    nb_clips_batch = []
    for video_ind in range(nb_in_batch):
        video_id = batch_video_id[video_ind].numpy().decode("utf-8")
        nb_clips = df.loc[df['video_id'] == video_id]['length'].values
        nb_clips_batch.append(nb_clips[0])
    return tf.convert_to_tensor(nb_clips_batch, dtype=tf.uint8)


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

    path_to_csv_train = config_dict['data_path'] + config_dict['train_video_lengths_folder'] + 'summary.csv'
    df_video_lengths_train = pd.read_csv(path_to_csv_train)
    path_to_csv_val = config_dict['data_path'] + config_dict['val_video_lengths_folder'] + 'summary.csv'
    df_video_lengths_val = pd.read_csv(path_to_csv_val)

    train_steps = len([sample for sample in train_dataset])
    val_steps = len([sample for sample in val_dataset])
    epochs_not_improved = 0

    @tf.function(input_signature=(tf.TensorSpec(shape=[config_dict['video_batch_size'], None, 20480], dtype=tf.float32),
                                  tf.TensorSpec(shape=[config_dict['video_batch_size'], None, 2], dtype=tf.float32),
                                  tf.TensorSpec(shape=[config_dict['video_batch_size'], None, 2], dtype=tf.uint8),
                                  tf.TensorSpec(shape=[config_dict['video_batch_size'],], dtype=tf.uint8)))
    def train_step(x, preds, y, lengths):
        with tf.GradientTape() as tape:
            if config_dict['video_loss'] == 'cross_entropy':
                preds = model([x, preds], training=True)
                y = y[:, 0, :]
                loss = loss_fn(y, preds)
            if config_dict['video_loss'] == 'mil':
                preds_seq = model([x, preds], training=True)
                sparse_loss, tv_p, tv_np, mil = get_sparse_pain_loss(y, preds_seq, lengths, config_dict)
                loss = sparse_loss
                preds_mil = test_and_eval.evaluate_sparse_pain(y, preds_seq, lengths, config_dict)
                preds = preds_mil
                y = y[:, 0, :]
            if config_dict['video_loss'] == 'mil_ce':
                preds_seq, preds_one = model([x, preds], training=True)
                y_one = y[:, 0, :]
                ce_loss = loss_fn(y_one, preds_one)
                sparse_loss, tv_p, tv_np, mil  = get_sparse_pain_loss(y, preds_seq, lengths, config_dict)
                preds_mil = test_and_eval.evaluate_sparse_pain(y, preds_seq, lengths, config_dict)
                preds = preds_mil
                y = y[:, 0, :]
                loss = ce_loss + sparse_loss
        grads = tape.gradient(loss, model.trainable_weights)
        grads_names = [tw.name for tw in model.trainable_weights]
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y, preds)
        return grads, grads_names, loss, tv_p, tv_np, mil

    @tf.function(input_signature=(tf.TensorSpec(shape=[config_dict['video_batch_size'], None, 20480], dtype=tf.float32),
                                  tf.TensorSpec(shape=[config_dict['video_batch_size'], None, 2], dtype=tf.float32),
                                  tf.TensorSpec(shape=[config_dict['video_batch_size'], None, 2], dtype=tf.uint8),
                                  tf.TensorSpec(shape=[config_dict['video_batch_size'],], dtype=tf.uint8)))
    def validation_step(x, preds, y, lengths):
        if config_dict['video_loss'] == 'cross_entropy':
            preds = model([x, preds], training=False)
            y = y[:, 0, :]
            loss = loss_fn(y, preds)
        if config_dict['video_loss'] == 'mil':
            preds_seq = model([x, preds], training=True)
            sparse_loss, tv_p, tv_np, mil = get_sparse_pain_loss(y, preds_seq, lengths, config_dict)
            loss = sparse_loss
            preds_mil = test_and_eval.evaluate_sparse_pain(y, preds_seq, lengths, config_dict)
            preds = preds_mil
            y = y[:, 0, :]
        if config_dict['video_loss'] == 'mil_ce':
            preds_seq, preds_one = model([x, preds], training=True)
            preds_one = tf.keras.layers.Activation('softmax')(preds_one)
            y_one = y[:, 0, :]
            ce_loss = loss_fn(y_one, preds_one)
            sparse_loss, tv_p, tv_np, mil  = get_sparse_pain_loss(y, preds_seq, lengths, config_dict)
            loss = ce_loss + sparse_loss
            preds_mil = test_and_eval.evaluate_sparse_pain(y, preds_seq, lengths, config_dict)
            preds = 1/2 * (preds_one + preds_mil)
            y = y[:, 0, :]
        val_acc_metric.update_state(y, preds)
        return loss, tv_p, tv_np, mil

    for epoch in range(config_dict['video_nb_epochs']):
        print('\nStart of epoch %d' % (epoch,))
        wandb.log({'epoch': epoch})
        start_time = time.time()

        with tqdm(total=train_steps) as pbar:
            for step, sample in enumerate(train_dataset):
                # if step > train_steps:
                #     break
                # step_start_time = time.time()
                pbar.update(1)
                feats_batch, preds_batch, labels_batch, video_id = sample
                
                lengths_batch = get_nb_clips_per_video(video_id, df_video_lengths_train)

                # print('\n Video ID: ', video_id)
                grads, grads_names, loss_value, tv_p, tv_np, mil = train_step(
                    feats_batch, preds_batch, labels_batch, lengths_batch)
                # import ipdb; ipdb.set_trace()
                # step_time = time.time() - step_start_time
                # print('Step time: %.2f' % step_time)
                wandb.log({'train_loss': loss_value.numpy()})
                wandb.log({'tv_nopain': tv_np.numpy()})
                wandb.log({'tv_pain': tv_p.numpy()})
                wandb.log({'mil': mil.numpy()})
                for ind, g in enumerate(grads):
                    wandb.log({grads_names[ind]: wandb.Histogram(g)})

                if step % config_dict['print_loss_every'] == 0:
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
                    lengths_batch = get_nb_clips_per_video(video_id, df_video_lengths_val)
                    # print('\n Video ID: ', video_id)
                    loss_value, tv_p, tv_np, mil = validation_step(
                        feats_batch, preds_batch, labels_batch, lengths_batch)
                    # step_time = time.time() - step_start_time
                    # print('Step time: %.2f' % step_time)

            wandb.log({'val_loss': loss_value.numpy()})
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
    for epoch in range(config_dict['nb_epochs']):
        print('\nStart of epoch %d' % (epoch,))
        wandb.log({'epoch': epoch})
        start_time = time.time()

        with tqdm(total=train_steps) as pbar:
            for step, sample in enumerate(train_dataset):
                if step > train_steps:
                    break
                # step_start_time = time.time()
                pbar.update(1)
                # x_batch_train, y_batch_train = sample
                x_batch_train, y_batch_train, paths = sample
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
                    # x_batch_val, y_batch_val = sample
                    x_batch_val, y_batch_val, paths = sample
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

    # @tf.function
    def get_features_step(x):
        predictions, features = model(x, training=False)
        # Downsample further with one MP layer, strides and kernel 2x2
        # The result per frame is 4x4x32.
        # features = tf.keras.layers.TimeDistributed(
        #     tf.keras.layers.MaxPool2D())(features)
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
    np.savez_compressed(config_dict['checkpoint'][:18] + '_saved_features_20480dims', features_to_save)


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

