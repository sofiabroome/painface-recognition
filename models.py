from keras.layers import Convolution2D, MaxPooling2D, LSTM, Dense
from keras.optimizers import Adam, Adagrad
from keras.models import Sequential


class Model:
    def __init__(self, name, input_shape, nb_conv_filters, nb_lstm_units,
                 kernel_size, nb_labels, lr, optimizer):
        """
        A class to build the preferred model.
        :param name: str | The name of the model
        :param input_shape: (int, int, int) | The shape of the data fed into the model.
        :param nb_conv_filters: int | The number of filters of the convolution, aka
                                      the output dimension from the convolution.
        :param nb_lstm_units: int | The number of LSTM cells.
        :param kernel_size: (int, int) | The size of the convolutional kernel.
        :param nb_labels: int | The number of possible classes to predict.
        :param lr: float | The learning rate during training.
        :param optimizer: str | The name of the optimizer (Adam or Adagrad, here).
        """
        self.name = name
        self.input_shape = input_shape
        self.nb_filters = nb_conv_filters
        self.nb_lstm_units = nb_lstm_units
        self.kernel_size = kernel_size
        self.nb_labels = nb_labels

        if self.name == 'conv_lstm':
            model = self.conv_lstm()

        metrics = ['accuracy']

        # Compile the network.
        if optimizer == 'adam':
            optimizer = Adam(lr=lr)
        else:
            optimizer = Adagrad(lr=lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

    def conv_lstm(self):
        model = Sequential()
        model.add(Convolution2D(input_shape=self.input_shape, filters=self.nb_conv_filters,
                                kernel_size=self.kernel_size))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(LSTM(self.nb_lstm_units))
        model.add(Dense(self.nb_labels, activation='softmax'))
        return model


