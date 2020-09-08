from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from wandb.keras import WandbCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(model_instance, config_dict, train_steps, val_steps, val_fraction,
          generator=None, val_generator=None, X_train=None, y_train=None):
    """
    Train the model.
    :param model_instance: Model object from my file models.py | The model instance.
                           model_instance.model is the keras Sequential()-object.
    :param args: [mixed types] | Command line args
    :param train_steps: int
    :param val_steps: int
    :param val_fraction: float
    :param X_train: np.ndarray
    :param y_train: np.ndarray
    :param generator: Generator object
    :param val_generator: Generator object
    :return: keras.Sequential() object | The (trained) model instance
    """
    print(model_instance.model.summary())

    best_model_path = create_best_model_path(model_instance, config_dict)
    print('best model path:')
    print(best_model_path)

    if config_dict['model'] == 'inception_4d_input':
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=config_dict['early_stopping'])
        checkpointer = ModelCheckpoint(filepath=best_model_path,
                                       monitor="val_loss",
                                       verbose=1,
                                       save_best_only=True,
                                       mode='min')
    else:
        early_stopping = EarlyStopping(monitor='val_binary_accuracy',
                                       patience=config_dict['early_stopping'])
        checkpointer = ModelCheckpoint(filepath=best_model_path,
                                       monitor="val_binary_accuracy",
                                       verbose=1,
                                       save_best_only=True,
                                       mode='max')

    catacc_test_history = CatAccTestHistory()
    catacc_train_history = CatAccTrainHistory()
    binacc_train_history = BinAccTrainHistory()
    binacc_test_history = BinAccTestHistory()
    pb = PrintBatch()

    if generator:
        print("TRAIN STEPS:")
        print(train_steps)
        print("VAL STEPS:")
        print(val_steps)

        if config_dict['model'] == 'inception_4d_input':
            # let's visualize layer names and layer indices to see how many layers
            # we should freeze:
            # for i, layer in enumerate(model_instance.base_model.layers):
            #    print(i, layer.name)
            for layer in model_instance.model.layers[:249]:
                layer.trainable = False
            for layer in model_instance.model.layers[249:]:
               layer.trainable = True

            from keras.optimizers import SGD
            model_instance.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                                                       loss='binary_crossentropy')
            # print('Model summary after unfreezing the layers after 249') 
            # print(model_instance.model.summary())
            model_instance.model.fit_generator(generator=generator,
                                               steps_per_epoch=train_steps,
                                               epochs=config_dict['nb_epochs'],
                                               callbacks=[early_stopping, checkpointer,
                                                          binacc_test_history, binacc_train_history,
                                                          WandbCallback(monitor='val_acc')],
                                               validation_data=val_generator,
                                               validation_steps=val_steps,
                                               verbose=1,
                                               workers=config_dict['nb_workers'])
        else:

            model_instance.model.fit_generator(generator=generator,
                                               steps_per_epoch=train_steps,
                                               epochs=config_dict['nb_epochs'],
                                               callbacks=[early_stopping, checkpointer,
                                                          binacc_test_history, binacc_train_history,
                                                          WandbCallback(monitor='val_acc')],
                                               validation_data=val_generator,
                                               validation_steps=val_steps,
                                               verbose=1,
                                               workers=config_dict['nb_workers'])

    else:
        if config_dict['round_to_batch']:
            X_train, y_train, X_val, y_val = val_split(X_train, y_train, val_fraction, config_dict['batch_size'])
            X_train = round_to_batch_size(X_train, config_dict['batch_size'])
            y_train = round_to_batch_size(y_train, config_dict['batch_size'])

            print(X_train.shape)
            print(y_train.shape)

            model_instance.model.fit(X_train, y_train,
                                     epochs=config_dict['nb_epochs'],
                                     shuffle=False,
                                     batch_size=config_dict['batch_size'],
                                     validation_data=(X_val, y_val),
                                     callbacks=[early_stopping, checkpointer,
                                                catacc_test_history, catacc_train_history,
                                                WandbCallback(monitor='val_acc')])

        else:
            model_instance.model.fit(X_train, y_train,
                                     epochs=config_dict['nb_epochs'],
                                     shuffle=False,
                                     batch_size=config_dict['batch_size'],
                                     validation_split=val_fraction,
                                     callbacks=[early_stopping, checkpointer,
                                                catacc_test_history, catacc_train_history,
                                                WandbCallback(monitor='val_acc')])

    plot_training(binacc_test_history, binacc_train_history, config_dict)

    return best_model_path


def val_split(X_train, y_train, val_fraction, batch_size, round_to_batch=True):
    if round_to_batch:
        ns = X_train.shape[0]
        num_val = int(val_fraction * ns - val_fraction * ns % batch_size)

        X_val = X_train[-num_val:, :]
        y_val = y_train[-num_val:]

        X_train = X_train[:-num_val, :]
        y_train = y_train[:-num_val:]

        # If pre-divided into batches...temp thing
        # X_val = X_train[-1:]
        # y_val = y_train[-1:]
        # X_train = X_train[:-1]
        # y_train = y_train[:-1]

    return X_train, y_train, X_val, y_val


def round_to_batch_size(data_array, batch_size):
    num_rows = data_array.shape[0]
    surplus = num_rows % batch_size
    data_array_rounded = data_array[:num_rows-surplus]
    return data_array_rounded


def create_best_model_path(model, config_dict):
    model_path = "models/BEST_MODEL_" + model.name + "_" + str(config_dict['optimizer']) +\
                 "_LSTMunits_" + str(model.nb_lstm_units) + "_CONVfilters_" + str(model.nb_conv_filters) +\
                 "_" + config_dict['job_identifier'] + ".h5"
    return model_path


class CatAccTestHistory(Callback):
    # def __init__(self):
    #     self.cataccs = []

    def on_train_begin(self, logs={}):
        self.cataccs = []

    def on_epoch_end(self, epoch, logs={}):
        self.cataccs.append(logs.get('val_categorical_accuracy'))


class CatAccTrainHistory(Callback):
    # def __init__(self):
    #     self.cataccs = []

    def on_train_begin(self, logs={}):
        self.cataccs = []

    def on_epoch_end(self, epoch, logs={}):
        self.cataccs.append(logs.get('categorical_accuracy'))


class BinAccTestHistory(Callback):
    # def __init__(self):
    #     self.cataccs = []

    def on_train_begin(self, logs={}):
        self.binaccs = []

    def on_epoch_end(self, epoch, logs={}):
        self.binaccs.append(logs.get('val_binary_accuracy'))


class BinAccTrainHistory(Callback):
    # def __init__(self):
    #     self.cataccs = []

    def on_train_begin(self, logs={}):
        self.binaccs = []

    def on_epoch_end(self, epoch, logs={}):
        self.binaccs.append(logs.get('binary_accuracy'))


class PrintBatch(Callback):
    def on_batch_end(self, epoch, logs={}):
        print(logs)


def plot_training(test_history,
                  train_history,
                  config_dict):
    plt.plot(test_history.binaccs, label='Validation set, categorical accuracy')
    plt.plot(train_history.binaccs, label='Training set, categorical accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(config_dict['model'] + "_" + config_dict['job_identifier'] + "_LSTM_UNITS_" +\
                str(config_dict['nb_lstm_units']) + "_CONV_FILTERS_" +\
                str(config_dict['nb_conv_filters']) + ".png")
    plt.close()

