from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from matplotlib import pyplot as plt


def train(model, args, X_train, y_train, X_val, y_val,
          batch_size, nb_epochs, early_stopping_patience, generator=None):
    """
    Train the model.
    :param model: keras.Sequential() object | The model instance
    :param args: [mixed types] | Command line args
    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :param batch_size: int | How many images per batch
    :param nb_epochs: int | (Maximum) number of training epochs
    :param early_stopping_patience: int | Number of epochs with no validation set
                                          improvement before early stopping.s
    :param generator: A generator, provided if the training data can't fit into memory.
    :return: keras.Sequential() object | The (trained) model instance
    """
    print(model.summary())
    best_model_path = create_best_model_path(model, args)
    early_stopping = EarlyStopping(monitor='val_categorical_accuracy',
                                   patience=early_stopping_patience)
    checkpointer = ModelCheckpoint(filepath=best_model_path,
                                   monitor="val_categorical_accuracy",
                                   verbose=1,
                                   save_best_only=True,
                                   mode='max')

    catacc_test_history = CatAccTestHistory()
    catacc_train_history = CatAccTrainHistory()

    if generator:
        model.fit_generator(generator=generator)
    else:
        model.fit(X_train, y_train,
                  epochs=nb_epochs,
                  shuffle=False,
                  batch_size=batch_size,
                  validation_data=(X_val, y_val),
                  callbacks=[early_stopping, checkpointer,
                             catacc_test_history, catacc_train_history])


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
    model_path = "models/BEST_MODEL_" + model.name + "_" + model.optimizer +\
                 "_WS_" + "_units_" + str(model.nb_hidden) + "_" + args.image_identifier + ".h5"
    return model_path


class CatAccTestHistory(Callback):
    def __init__(self):
        self.cataccs = []

    # def on_train_begin(self, logs={}):
    #     self.cataccs = []

    def on_epoch_end(self, epoch, logs={}):
        self.cataccs.append(logs.get('val_categorical_accuracy'))


class CatAccTrainHistory(Callback):
    def __init__(self):
        self.cataccs = []

    # def on_train_begin(self, logs={}):
    #     self.cataccs = []

    def on_epoch_end(self, epoch, logs={}):
        self.cataccs.append(logs.get('categorical_accuracy'))
