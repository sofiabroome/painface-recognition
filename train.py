from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import wandb
import time
import os

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
    best_model_path = create_best_model_path(model_instance, config_dict)
    print('Saving best epoch model to: ', best_model_path)

    print("Setting the optimizer to {}.".format(config_dict['optimizer']))
    if config_dict['optimizer'] == 'adam':
        optimizer = Adam(lr=config_dict['lr'])
    if config_dict['optimizer'] == 'adadelta':
        optimizer = Adadelta(lr=config_dict['lr'])
    if config_dict['optimizer'] == 'adagrad':
        optimizer = Adagrad(lr=config_dict['lr'])

    print("Using binary crossentropy and binary accuracy metrics.")
    model_instance.model.compile(loss='binary_crossentropy',
                                 optimizer=optimizer,
                                 metrics=['binary_accuracy'])

    if config_dict['train_mode'] == 'keras':
        keras_train(model_instance.model, best_model_path,
                    config_dict, train_steps, val_steps,
                    train_dataset, val_dataset)

    if config_dict['train_mode'] == 'low_level':
        low_level_train(model_instance.model, best_model_path,
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


def low_level_train(model, ckpt_path, optimizer, config_dict,
                    train_steps, val_steps=None,
                    train_dataset=None, val_dataset=None):

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_old = 0

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
                x_batch_train, y_batch_train = sample
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

            with tqdm(total=train_steps) as pbar:
                for step, sample in enumerate(val_dataset):
                    if step > val_steps:
                        break
                    pbar.update(1)
                    # step_start_time = time.time()
                    x_batch_val, y_batch_val = sample
                    loss_value = validation_step(x_batch_val, y_batch_val)
                    # step_time = time.time() - step_start_time
                    # print('Step time: %.2f' % step_time)
                    wandb.log({'val_loss': loss_value.numpy()})

            val_acc = val_acc_metric.result()
            print("Validation acc: %.4f" % (float(val_acc),))

            if val_acc > val_acc_old:
                print('The validation acc improved, saving checkpoint...')
                model.save_weights(ckpt_path)
                val_acc_old = val_acc
            val_acc_metric.reset_states()
        else:
            print('\n Not validating but saving '
                  'checkpoint after epoch {}'.format(epoch))
            model.save_weights(ckpt_path)
        print("Epoch time taken: %.2fs" % (time.time() - start_time))


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


def create_best_model_path(model, config_dict):
    model_path = '_'.join(('models/best_model',
                          model.name,
                          config_dict['job_identifier'])) + '.ckpt'
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

