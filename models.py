from keras.layers import Convolution2D, Convolution3D, MaxPooling2D, MaxPooling3D, LSTM, Dense, Flatten
from keras.layers import ZeroPadding3D, Dropout, BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, Adagrad
from keras.applications import InceptionV3
from keras.models import Sequential
from keras import backend as K

# K.set_image_dim_ordering('th')


class Model:
    def __init__(self, args):
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
        self.name = args.model
        self.input_shape = (args.input_width, args.input_height)
        self.nb_conv_filters = args.nb_conv_filters
        self.nb_lstm_units = args.nb_lstm_units
        self.kernel_size = args.kernel_size
        self.nb_labels = args.nb_labels
        self.dropout_2 = args.dropout_2
        self.dropout_1 = args.dropout_1
        self.seq_length = args.seq_length
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.batch_size = args.batch_size
        self.nb_lstm_layers = args.nb_lstm_layers
        self.nb_dense_units = args.nb_dense_units

        if self.name == 'conv2d_timedist_lstm':
            print("Conv2d-lstm model timedist")
            self.model = self.conv2d_timedist_lstm()

        if self.name == 'conv2d_timedist_lstm_stateful':
            print("Stateful timedist conv2d-lstm model")
            self.model = self.conv2d_timedist_lstm_stateful()

        if self.name == 'conv2d_lstm':
            print("Conv2d-lstm model")
            self.model = self.conv2d_lstm()

        if self.name == 'conv2d_lstm_informed':
            print("Conv2d-lstm model informed")
            self.model = self.conv2d_lstm_informed()

        if self.name == 'conv2d_lstm_stateful':
            print("Conv2d-lstm model stateful")
            self.model = self.conv2d_lstm_stateful()

        if self.name == 'conv3d_informed':
            print('Conv3D Informed')
            self.model = self.conv3d_informed()

        if self.name == 'inception_lstm_5d_input':
            print('inception_lstm_5d_input')
            self.model = self.inception_lstm_5d_input()

        if self.name == 'inception_lstm_4d_input':
            print('inception_lstm_4d_input')
            self.model = self.inception_lstm_4d_input()


        if self.optimizer == 'adam':
            optimizer = Adam(lr=self.lr)
        else:
            print("Setting the optimizer to Adagrad.")
            optimizer = Adagrad(lr=lr)

        # Compile the network.
        # print("Using categorical crossentropy and categorical accuracy metrics.")
        # self.model.compile(loss='categorical_crossentropy',
        #                    optimizer=optimizer,
        #                    metrics=['categorical_accuracy'])

        print("Using binary crossentropy and binary accuracy metrics.")
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=['binary_accuracy'])

    def conv2d_lstm(self):
        model = Sequential()
        model.add(Convolution2D(filters=self.nb_conv_filters,
                                kernel_size=(self.kernel_size, self.kernel_size),
                                input_shape=(self.input_shape[0], self.input_shape[1], 3),
                                batch_input_shape=(None, self.input_shape[0], self.input_shape[1], 3),
                                activation='relu'))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(3, 3),
                                activation='relu'))
        model.add(BatchNormalization())
        model.add(TimeDistributed(Flatten()))
        if self.nb_lstm_layers == 1:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers == 2:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers == 3:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers == 4:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_labels == 2:
            print("2 labels, using sigmoid activation instead of softmax.")
            model.add(Dense(self.nb_labels, activation='sigmoid'))
        else:
            model.add(Dense(self.nb_labels, activation='softmax'))
        return model

    def conv2d_lstm_informed(self):
        model = Sequential()
        model.add(Convolution2D(filters=self.nb_conv_filters,
                                kernel_size=(self.kernel_size, self.kernel_size),
                                input_shape=(self.input_shape[0], self.input_shape[1], 3),
                                batch_input_shape=(None, self.input_shape[0], self.input_shape[1], 3),
                                activation='relu'))
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(self.dropout_1))
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(self.dropout_1))
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(self.dropout_1))
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(BatchNormalization())
        model.add(TimeDistributed(Flatten()))
        if self.nb_lstm_layers == 1:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers == 2:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers == 3:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers == 4:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
            model.add(Dense(self.nb_dense_units, activation='relu'))
            model.add(Dropout(self.dropout_2))
        if self.nb_labels == 2:
            print("2 labels, using sigmoid activation instead of softmax.")
            model.add(Dense(self.nb_labels, activation='sigmoid'))
        else:
            model.add(Dense(self.nb_labels, activation='softmax'))
        return model

    def conv2d_lstm_stateful(self):
        model = Sequential()
        model.add(Convolution2D(filters=self.nb_conv_filters,
                                kernel_size=(self.kernel_size, self.kernel_size),
                                input_shape=(self.input_shape[0], self.input_shape[1], 3),
                                batch_input_shape=(self.batch_size, self.input_shape[0], self.input_shape[1], 3),
                                activation='relu'))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(3, 3),
                                activation='relu'))
        model.add(TimeDistributed(Flatten()))
        model.add((LSTM(self.nb_lstm_units,
                        stateful=True,
                        dropout=self.dropout_2,
                        input_shape=(None, self.seq_length, None),
                        return_sequences=False,
                        implementation=2)))
        if self.nb_labels == 2:
            print("2 labels, using sigmoid activation instead of softmax.")
            model.add(Dense(self.nb_labels, activation='sigmoid'))
        else:
            model.add(Dense(self.nb_labels, activation='softmax'))
        return model

    def inception_lstm_4d_input(self):
        model = Sequential()
        model.add(InceptionV3(include_top=False, input_shape=(self.input_shape[0],
                                                              self.input_shape[1],
                                                              3)))
        model.add(TimeDistributed(Flatten()))
        model.add((LSTM(self.nb_lstm_units,
                        stateful=False,
                        dropout=self.dropout_2,
                        input_shape=(None, self.seq_length, None),
                        return_sequences=False,
                        implementation=2
                        )))
        if self.nb_labels == 2:
            print("2 labels, using sigmoid activation instead of softmax.")
            model.add(Dense(self.nb_labels, activation='sigmoid'))
        else:
            model.add(Dense(self.nb_labels, activation='softmax'))
        return model

    def inception_lstm_5d_input(self):
        model = Sequential()
        model.add(InceptionV3(include_top=False, input_shape=(self.seq_length,
                                                              self.input_shape[0],
                                                              self.input_shape[1],
                                                              3)))
        model.add(TimeDistributed()())
        model.add(Convolution3D(self.nb_conv_filters, 3, 3, 3))
        model.add(Convolution3D(self.nb_of_filters, 3, 3, 3))
        model.add(Flatten())
        model.add(Dense(self.nb_labels, activation="softmax"))
        return model

    def conv3d_informed(self):
        model = Sequential()
        # 1st layer group
        model.add(Convolution3D(filters=self.nb_conv_filters,
                                kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                                activation='relu',
                                input_shape=(self.seq_length, self.input_shape[0], self.input_shape[1], 3),
                                batch_input_shape=(None, self.seq_length, self.input_shape[0], self.input_shape[1], 3)))
        model.add(Convolution3D(filters=self.nb_conv_filters,
                                kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(MaxPooling3D())
        model.add(Dropout(self.dropout_1))
        model.add(BatchNormalization)
        model.add(Convolution3D(filters=self.nb_conv_filters,
                                kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(Convolution3D(filters=self.nb_conv_filters,
                                kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                                activation='relu'))
        model.add(MaxPooling3D())
        model.add(Dropout(self.dropout_1))
        model.add(BatchNormalization)
        model.add(Flatten())
        model.add(Dense(self.nb_dense_units, activation='relu'))
        model.add(Dropout(self.dropout_2))
        model.add(Dense(self.nb_labels, activation="softmax"))

        return model

    def conv2d_timedist_lstm(self):
        model = Sequential()
        model.add(TimeDistributed(Convolution2D(filters=self.nb_conv_filters,
                                                kernel_size=(self.kernel_size,
                                                             self.kernel_size)),
                                                input_shape=(self.seq_length,
                                                             self.input_shape[0],
                                                             self.input_shape[1], 3),
                                                batch_input_shape=(None, self.seq_length,
                                                                   self.input_shape[0],
                                                                   self.input_shape[1], 3)))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(TimeDistributed(Flatten()))
        model.add((LSTM(self.nb_lstm_units,
                        stateful=False,
                        dropout=self.dropout_2,
                        input_shape=(None, self.seq_length, None),
                        return_sequences=True,
                        implementation=2)))
        model.add(TimeDistributed(Dense(self.nb_labels, activation='softmax')))
        return model

    def conv2d_timedist_lstm_stateful(self):
        model = Sequential()
        model.add(TimeDistributed(Convolution2D(filters=self.nb_conv_filters,
                                                kernel_size=(self.kernel_size,
                                                             self.kernel_size)),
                                                input_shape=(self.seq_length,
                                                             self.input_shape[0],
                                                             self.input_shape[1], 3),
                                                batch_input_shape=(self.batch_size,
                                                                   self.seq_length,
                                                                   self.input_shape[0],
                                                                   self.input_shape[1], 3)))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(TimeDistributed(Flatten()))
        model.add((LSTM(self.nb_lstm_units,
                        stateful=True,
                        dropout=self.dropout_2,
                        input_shape=(None, self.seq_length, None),
                        return_sequences=True,
                        implementation=2)))
        model.add(TimeDistributed(Dense(self.nb_labels, activation='softmax')))
        return model
