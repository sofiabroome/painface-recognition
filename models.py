from keras.layers import Convolution2D, MaxPooling2D, MaxPooling3D, LSTM, Dense, Flatten
from keras.layers import ZeroPadding3D, Dropout, BatchNormalization, concatenate, Input, Conv3D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, Adagrad
from keras.applications import InceptionV3
from keras.models import Sequential, Model
from keras import regularizers
from keras import backend as K

# K.set_image_dim_ordering('th')


class MyModel:
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
        self.input_shape = (args.input_height, args.input_width)
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
            self.model = self.conv2d_lstm(channels=3)

        if self.name == 'conv2d_lstm_stateful':
            print("Conv2d-lstm model stateful")
            self.model = self.conv2d_lstm_stateful()

        if self.name == 'conv2d_lstm_informed':
            print("Conv2d-lstm model informed")
            self.model = self.conv2d_lstm_informed()

        if self.name == 'conv3d_informed':
            print('Conv3D Informed')
            self.model = self.conv3d_informed()

        if self.name == 'inception_lstm_5d_input':
            print('inception_lstm_5d_input')
            self.model = self.inception_lstm_5d_input()

        if self.name == 'inception_lstm_4d_input':
            print('inception_lstm_4d_input')
            self.model = self.inception_lstm_4d_input()

        if self.name == '2stream':
            print('2stream')
            self.model = self.two_stream()

        if self.name == '2stream_5d':
            print('2stream 5D')
            self.model = self.two_stream_5d()

        if self.name == '2stream_stateful':
            print('2stream stateful')
            self.model = self.two_stream_stateful()

        if self.name == '2stream_pretrained':
            print('2stream_pretrained')
            self.model = self.two_stream_pretrained()

        if self.name == 'simonyan':
            print('Simonyan')
            self.model = self.simonyan(channels=3)


        if self.optimizer == 'adam':
            optimizer = Adam(lr=self.lr)
        else:
            print("Setting the optimizer to Adagrad.")
            optimizer = Adagrad(lr=self.lr)

        # Compile the network.
        # print("Using categorical crossentropy and categorical accuracy metrics.")
        # self.model.compile(loss='categorical_crossentropy',
        #                    optimizer=optimizer,
        #                    metrics=['categorical_accuracy'])

        print("Using binary crossentropy and binary accuracy metrics.")
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=['binary_accuracy'])

    def two_stream_pretrained(self):
        # Functional API
        rgb_model = self.conv2d_lstm(channels=3, top_layer=False)
        image_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))
        encoded_image = rgb_model(image_input)

        of_model = InceptionV3(include_top=False)
        flatten = TimeDistributed(Flatten())(of_model)
        lstm_layer = LSTM(self.nb_lstm_units,
                          return_sequences=False,
                          dropout=self.dropout_2)(flatten)
        of_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))
        encoded_of = lstm_layer(of_input)

        merged = concatenate([encoded_image, encoded_of], axis=-1)

        if self.nb_labels == 2:
            output = Dense(self.nb_labels, activation='sigmoid')(merged)
        else:
            output = Dense(self.nb_labels, activation='softmax')(merged)

        two_stream_model = Model(inputs=[image_input, of_input], outputs=[output])
        return two_stream_model

    def two_stream(self):
        # Functional API
        rgb_model = self.conv2d_lstm(channels=3, top_layer=False)
        image_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))
        encoded_image = rgb_model(image_input)

        of_model = self.conv2d_lstm(channels=3, top_layer=False)
        of_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))
        encoded_of = of_model(of_input)

        merged = concatenate([encoded_image, encoded_of], axis=-1)
        # dense = Dense(self.nb_dense_units, activation='relu')(merged)

        if self.nb_labels == 2:
            output = Dense(self.nb_labels, activation='sigmoid')(merged)
        else:
            output = Dense(self.nb_labels, activation='softmax')(merged)

        two_stream_model = Model(inputs=[image_input, of_input], outputs=[output])
        return two_stream_model

    def two_stream_5d(self):
        # Functional API
        rgb_model = TimeDistributed(self.conv2d_lstm(channels=3, top_layer=False, stateful=False))
        # rgb_model = TimeDistributed(self.simonyan(channels=3, top_layer=False, stateful=False))
        image_input = Input(shape=(None, self.input_shape[0], self.input_shape[1], 3))
        encoded_image = rgb_model(image_input)

        of_model = TimeDistributed(self.conv2d_lstm(channels=3, top_layer=False, stateful=False))
        # of_model = TimeDistributed(self.simonyan(channels=3, top_layer=False, stateful=False))
        of_input = Input(shape=(None, self.input_shape[0], self.input_shape[1], 3))
        encoded_of = of_model(of_input)

        merged = concatenate([encoded_image, encoded_of], axis=-1)
        merged = Dropout(.2)(merged)
        # dense = Dense(self.nb_dense_units, activation='relu')(merged)

        if self.nb_labels == 2:
            output = Dense(self.nb_labels, activation='sigmoid')(merged)
        else:
            output = Dense(self.nb_labels, activation='softmax')(merged)

        two_stream_model = Model(inputs=[image_input, of_input], outputs=[output])

        # # Functional API
        # rgb_model = self.conv3d_informed(channels=3, top_layer=False)
        # image_input = Input(shape=(None, self.input_shape[0], self.input_shape[1], 3),
        #                     batch_shape=(None, None, self.input_shape[0], self.input_shape[1], 3))
        # encoded_image = rgb_model(image_input)
        #
        # of_model = self.conv3d_informed(channels=3, top_layer=False)
        # of_input = Input(shape=(None, self.input_shape[0], self.input_shape[1], 3),
        #                  batch_shape=(None, None, self.input_shape[0], self.input_shape[1], 3))
        # encoded_of = of_model(of_input)
        #
        # merged = concatenate([encoded_image, encoded_of], axis=-1)
        # # dense = Dense(self.nb_dense_units, activation='relu')(merged)
        # # mtd = TimeDistributed(merged)
        # if self.nb_labels == 2:
        #     dense = Dense(self.nb_labels, activation='sigmoid')(merged)
        #     output = TimeDistributed(dense)(merged)
        # else:
        #     output = Dense(self.nb_labels, activation='softmax')(merged)
        #
        # two_stream_model = Model(inputs=[image_input, of_input], outputs=[output])
        return two_stream_model

    def two_stream_stateful(self):
        # Functional API
        rgb_model = self.conv2d_lstm_stateful(channels=3, top_layer=False)
        image_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3),
                            batch_shape=(None, self.input_shape[0], self.input_shape[1], 3))
        encoded_image = rgb_model(image_input)

        of_model = self.conv2d_lstm_stateful(channels=3, top_layer=False)
        of_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3),
                         batch_shape=(None, self.input_shape[0], self.input_shape[1], 3))
        encoded_of = of_model(of_input)

        merged = concatenate([encoded_image, encoded_of], axis=-1)
        # dense = Dense(self.nb_dense_units, activation='relu')(merged)

        if self.nb_labels == 2:
            output = Dense(self.nb_labels, activation='sigmoid')(merged)
        else:
            output = Dense(self.nb_labels, activation='softmax')(merged)

        two_stream_model = Model(inputs=[image_input, of_input], outputs=[output])
        return two_stream_model

    def simonyan(self, channels, top_layer=True, stateful=False):
        model = Sequential()
        model.add(Convolution2D(filters=96,
                                kernel_size=(7,7),
                                kernel_initializer='he_uniform',
                                activation='relu',
                                input_shape=(self.input_shape[0], self.input_shape[1], channels),
                                batch_input_shape=(self.batch_size, self.input_shape[0],
                                                   self.input_shape[1], channels),
                                strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Convolution2D(filters=256,
                                kernel_size=(5, 5),
                                kernel_initializer='he_uniform',
                                activation='relu',
                                strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Convolution2D(filters=512,
                                kernel_size=(3, 3),
                                kernel_initializer='he_uniform',
                                activation='relu',
                                strides=(1, 1)))
        model.add(Convolution2D(filters=512,
                                kernel_size=(3, 3),
                                kernel_initializer='he_uniform',
                                activation='relu',
                                strides=(1, 1)))
        model.add(Convolution2D(filters=512,
                                kernel_size=(3, 3),
                                kernel_initializer='he_uniform',
                                activation='relu',
                                strides=(1, 1)))
        model.add(MaxPooling2D())
        model.add(Dense(4096))
        model.add(Dropout(0.5))
        model.add(Dense(2048))
        model.add(Dropout(0.5))
        if top_layer:
            if self.nb_labels == 2:
                model.add(Dense(self.nb_labels, activation='sigmoid'))
            else:
                model.add(Dense(self.nb_labels, activation='softmax'))
        return model

    def conv2d_lstm(self, channels, top_layer=True, stateful=False):
        model = Sequential()
        model.add(Convolution2D(filters=self.nb_conv_filters,
                                kernel_size=(self.kernel_size, self.kernel_size),
                                input_shape=(self.input_shape[0], self.input_shape[1], channels),
                                batch_input_shape=(self.batch_size, self.input_shape[0], self.input_shape[1], channels),
                                activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(3, 3),
                                activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(TimeDistributed(Flatten()))
        if self.nb_lstm_layers == 1:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=stateful,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            batch_input_shape=(self.batch_size, None, None, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers == 2:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=stateful,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=stateful,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers == 3:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=stateful,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=stateful,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=stateful,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers == 4:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=stateful,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=stateful,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=stateful,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=stateful,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if top_layer:
            if self.nb_labels == 2:
                print("2 labels, using sigmoid activation instead of softmax.")
                model.add(Dense(self.nb_labels, activation='sigmoid'))
            else:
                model.add(Dense(self.nb_labels, activation='softmax'))
        return model

    def conv2d_lstm_informed(self, top_layer=True):
        model = Sequential()
        model.add(Convolution2D(filters=self.nb_conv_filters,
                                kernel_size=(self.kernel_size, self.kernel_size),
                                input_shape=(self.input_shape[0], self.input_shape[1], 3),
                                batch_input_shape=(None, self.input_shape[0], self.input_shape[1], 3),
                                activation='relu', kernel_initializer='he_uniform'))
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D())
        model.add(Dropout(self.dropout_1))
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu', kernel_initializer='he_uniform'))
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D())
        model.add(Dropout(self.dropout_1))
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D())
        model.add(Dropout(self.dropout_1))
        model.add(BatchNormalization())
        model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
                                activation='relu', kernel_initializer='he_uniform'))
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
            model.add(Dense(self.nb_dense_units, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dropout(self.dropout_2))
        if top_layer:
            if self.nb_labels == 2:
                print("2 labels, using sigmoid activation instead of softmax.")
                model.add(Dense(self.nb_labels, activation='sigmoid'))
            else:
                model.add(Dense(self.nb_labels, activation='softmax'))
        return model

    def conv2d_lstm_stateful(self, channels=3, top_layer=True):
        model = Sequential()
        model.add(Convolution2D(filters=self.nb_conv_filters,
                                kernel_size=(self.kernel_size, self.kernel_size),
                                input_shape=(self.input_shape[0], self.input_shape[1], channels),
                                batch_input_shape=(self.batch_size, self.input_shape[0], self.input_shape[1], channels),
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
        if top_layer:
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
        model.add(Conv3D(self.nb_conv_filters, 3, 3, 3))
        model.add(Conv3D(self.nb_of_filters, 3, 3, 3))
        model.add(Flatten())
        model.add(Dense(self.nb_labels, activation="softmax"))
        return model

    def conv3d_informed(self, channels=3, top_layer=True):
        model = Sequential()
        # 1st layer group
        model.add(Conv3D(filters=self.nb_conv_filters,
                         kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                         activation='relu',
                         input_shape=(None, self.input_shape[0], self.input_shape[1], channels),
                         batch_input_shape=(None, self.input_shape[0], self.input_shape[1], channels),
                         data_format='channels_last'))
        model.add(Conv3D(filters=self.nb_conv_filters,
                         kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                         activation='relu'))
        model.add(MaxPooling3D())
        model.add(Dropout(self.dropout_1))
        model.add(BatchNormalization())
        model.add(Conv3D(filters=self.nb_conv_filters,
                         kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                         activation='relu'))
        # model.add(Conv3D(filters=self.nb_conv_filters,
        #                         kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
        #                         activation='relu'))
        model.add(MaxPooling3D())
        model.add(Dropout(self.dropout_1))
        model.add(BatchNormalization())
        model.add(TimeDistributed(Flatten()))
        # model.add(Flatten())
        if self.nb_lstm_layers == 1:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            return_sequences=True,
                            implementation=2)))
        if self.nb_lstm_layers == 2:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers == 3:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers == 4:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            return_sequences=False,
                            implementation=2)))
        # model.add(TimeDistributed(Dense(self.nb_dense_units, activation='relu')))
        # model.add((Dropout(self.dropout_2)))
        if top_layer:
            if self.nb_labels == 2:
                model.add(Dense(self.nb_labels, activation="sigmoid"))
            else:
                model.add(Dense(self.nb_labels, activation="softmax"))

        return model

    def conv2d_timedist_lstm(self):

        rgb_model = TimeDistributed(self.conv2d_lstm_stateful(channels=3, top_layer=False))
        image_input = Input(shape=(None, self.input_shape[0], self.input_shape[1], 3))
        encoded_image = rgb_model(image_input)

        if self.nb_labels == 2:
            output = Dense(self.nb_labels, activation='sigmoid')(encoded_image)
        else:
            output = Dense(self.nb_labels, activation='softmax')(encoded_image)

        model = Model(inputs=image_input, outputs=output)

        # model = Sequential()
        # model.add(TimeDistributed(Convolution2D(filters=self.nb_conv_filters,
        #                                         kernel_size=(self.kernel_size,
        #                                                      self.kernel_size)),
        #                                         input_shape=(self.seq_length,
        #                                                      self.input_shape[0],
        #                                                      self.input_shape[1], 3),
        #                                         batch_input_shape=(None, self.seq_length,
        #                                                            self.input_shape[0],
        #                                                            self.input_shape[1], 3)))
        # model.add(TimeDistributed(MaxPooling2D()))
        # model.add(TimeDistributed(Flatten()))
        # model.add((LSTM(self.nb_lstm_units,
        #                 stateful=False,
        #                 dropout=self.dropout_2,
        #                 input_shape=(None, self.seq_length, None),
        #                 return_sequences=True,
        #                 implementation=2)))
        # model.add(TimeDistributed(Dense(self.nb_labels, activation='softmax')))
        return model

    def conv2d_timedist_lstm_stateful(self):
        model = Sequential()
        model.add(TimeDistributed(Convolution2D(filters=self.nb_conv_filters,
                                                kernel_size=(self.kernel_size,
                                                             self.kernel_size),
                                                activation='relu'),
                                  input_shape=(self.seq_length,self.input_shape[0],self.input_shape[1], 3),
                                  batch_input_shape=(self.batch_size, self.seq_length,
                                                     self.input_shape[0], self.input_shape[1], 3)))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(Dropout(self.dropout_1))
        model.add(BatchNormalization())
        model.add(TimeDistributed(Convolution2D(filters=self.nb_conv_filters,
                                                kernel_size=(self.kernel_size, self.kernel_size),
                                                activation='relu',
                                                kernel_initializer='he_uniform')))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(Dropout(self.dropout_1))
        model.add(BatchNormalization())
        model.add(TimeDistributed(Convolution2D(filters=self.nb_conv_filters,
                                                kernel_size=(self.kernel_size, self.kernel_size),
                                                activation='relu',
                                                kernel_initializer='he_uniform')))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(Dropout(self.dropout_1))
        model.add(BatchNormalization())
        model.add(TimeDistributed(Convolution2D(filters=self.nb_conv_filters,
                                                kernel_size=(self.kernel_size, self.kernel_size),
                                                activation='relu',
                                                kernel_initializer='he_uniform')))
        model.add(Dropout(self.dropout_1))
        model.add(BatchNormalization())
        model.add(TimeDistributed(Flatten()))
        model.add((LSTM(self.nb_lstm_units,
                        kernel_regularizer=regularizers.l1_l2(0.01,0.01),
                        stateful=True,
                        dropout=self.dropout_2,
                        input_shape=(None, self.seq_length, None),
                        return_sequences=True,
                        implementation=2)))
        model.add(TimeDistributed(Dense(self.nb_labels, activation='softmax')))
        return model
