from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from matplotlib import pyplot as plt
import tensorflow as tf


config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def train(model_instance, args, nb_train_samples, nb_val_samples, val_fraction,
          generator=None, val_generator=None, X_train=None, y_train=None):
    """
    Train the model.
    :param model_instance: Model object from my file models.py | The model instance.
                           model_instance.model is the keras Sequential()-object.
    :param args: [mixed types] | Command line args
    :param X_train: np.ndarray
    :param y_train: np.ndarray
    :param batch_size: int | How many images per batch
    :param nb_epochs: int | (Maximum) number of training epochs
    :param early_stopping_patience: int | Number of epochs with no validation set
                                          improvement before early stopping.
    :param generator: A generator, provided if the training data can't fit into memory.
    :return: keras.Sequential() object | The (trained) model instance
    """
    print(model_instance.model.summary())

    best_model_path = create_best_model_path(model_instance, args)
    print('best model path:')
    print(best_model_path)
    # Think about: choose between binary or categorical accuracy.
    early_stopping = EarlyStopping(monitor='val_binary_accuracy',
                                   patience=args.early_stopping)
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
        val_steps = int(nb_val_samples / args.batch_size)
        train_steps = int(nb_train_samples/args.batch_size)
        if args.test == 1:
            train_steps = 2
            val_steps = 2
        print("TRAIN STEPS:")
        print(train_steps)
        print("VAL STEPS:")
        print(val_steps)
        model_instance.model.fit_generator(generator=generator,
                                           steps_per_epoch= train_steps,
                                           epochs=args.nb_epochs,
                                           callbacks=[early_stopping, checkpointer,
                                                      binacc_test_history, binacc_train_history],
                                           validation_data=val_generator,
                                           validation_steps=val_steps,
                                           verbose=1)
    else:
        if args.round_to_batch:
            X_train, y_train, X_val, y_val = val_split(X_train, y_train, val_fraction, args.batch_size)
            X_train = round_to_batch_size(X_train, args.batch_size)
            y_train = round_to_batch_size(y_train, args.batch_size)

            print(X_train.shape)
            print(y_train.shape)

            model_instance.model.fit(X_train, y_train,
                                     epochs=args.nb_epochs,
                                     shuffle=False,
                                     batch_size=args.batch_size,
                                     validation_data=(X_val, y_val),
                                     callbacks=[early_stopping, checkpointer,
                                                catacc_test_history, catacc_train_history])

        else:
            model_instance.model.fit(X_train, y_train,
                                     epochs=args.nb_epochs,
                                     shuffle=False,
                                     batch_size=args.batch_size,
                                     validation_split=val_fraction,
                                     callbacks=[early_stopping, checkpointer,
                                                catacc_test_history, catacc_train_history])
    return model_instance.model


def val_split(X_train, y_train, val_fraction, batch_size, round_to_batch=True):
    if round_to_batch:
        ns = X_train.shape[0]
        num_val = int(val_fraction * ns - val_fraction * ns % batch_size)

        X_val = X_train[-num_val:, :]
        y_val = y_train[-num_val:]

        X_train = X_train[:-num_val, :]
        y_train = y_train[:-num_val:]
        import ipdb;ipdb.set_trace()

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


def plot_training(catacc_test_history,
                  catacc_train_history,
                  image_identifier,
                  ws,
                  model_name):
    plt.plot(catacc_test_history.cataccs, label='Val, categorical acc.')
    plt.plot(catacc_train_history.cataccs, label='Train, categorical acc.')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(model_name + '_' + image_identifier + "_WS_" + str(ws) + '.png')
    plt.close()


def create_best_model_path(model, args):
    model_path = "models/BEST_MODEL_" + model.name + "_" + str(args.optimizer) +\
                 "_LSTMunits_" + str(model.nb_lstm_units) + "_CONVfilters_" + str(model.nb_conv_filters) +\
                 "_" + args.image_identifier + ".h5"
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
