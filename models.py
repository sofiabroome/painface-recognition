from keras.layers import Convolution2D, MaxPooling2D, MaxPooling3D, LSTM, Dense, Flatten
from keras.layers import Dropout, BatchNormalization, concatenate, add, Input, Conv3D, multiply
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, Adagrad, Adadelta
from keras.applications import InceptionV3
from keras.models import Sequential, Model


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

        if self.name == 'conv3d_lstm':
            print("Conv3d-lstm model")
            self.model = self.conv3d_lstm(channels=3)

        if self.name == 'conv2d_lstm_5d':
            print("Conv2d-lstm model 5D")
            self.model = self.conv2d_lstm_5d(channels=3)

        if self.name == 'conv2d_lstm_stateful':
            print("Conv2d-lstm model stateful")
            self.model = self.conv2d_lstm_stateful()

        if self.name == 'conv2d_lstm_informed':
            print("Conv2d-lstm model informed")
            self.model = self.conv2d_lstm_informed()

        if self.name == 'inception_lstm_5d_input':
            print('inception_lstm_5d_input')
            self.model = self.inception_lstm_5d_input()

        if self.name == 'inception_lstm_4d_input':
            print('inception_lstm_4d_input')
            self.model = self.inception_lstm_4d_input()

        if self.name == 'inception_4d_input':
            print('inception_4d_input')
            self.model = self.inception_4d_input()

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

        if self.name == 'convolutional_LSTM':
            print('Convolutional LSTM (not fully connected)')
            self.model = self.convolutional_LSTM(channels=3)

        if self.optimizer == 'adam':
            optimizer = Adam(lr=self.lr)
        elif self.optimizer == 'adadelta':
            print("Setting the optimizer to Adadelta.")
            optimizer = Adadelta(lr=self.lr)
        else:
            print("Setting the optimizer to Adagrad.")
            optimizer = Adagrad(lr=self.lr)

        print("Using binary crossentropy and binary accuracy metrics.")
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=['binary_accuracy'])

    def conv2d_lstm_5d(self, channels=3):
        model = TimeDistributed(self.conv2d_lstm(channels=3, top_layer=False))
        image_input = Input(shape=(self.seq_length, self.input_shape[0], self.input_shape[1], channels))
        encoded_image = model(image_input)

        dropout = Dropout(.2)(encoded_image)

        if self.nb_labels == 2:
            output = Dense(self.nb_labels, activation='sigmoid')(dropout)
        else:
            output = Dense(self.nb_labels, activation='softmax')(dropout)

        whole_model = Model(inputs=image_input, outputs=output)
        return whole_model

    def two_stream_pretrained(self):
        # Functional API
        rgb_model = TimeDistributed(self.conv2d_lstm(channels=3, top_layer=False))
        image_input = Input(shape=(self.seq_length, self.input_shape[0], self.input_shape[1], 3))
        encoded_image = rgb_model(image_input)

        of_model = TimeDistributed(InceptionV3(include_top=False))

        of_input = Input(shape=(self.seq_length, self.input_shape[0], self.input_shape[1], 3))
        encoded_of = of_model(of_input)
        flatten = TimeDistributed(Flatten())(encoded_of)
        lstm_layer = LSTM(self.nb_lstm_units,
                          return_sequences=True,
                          dropout=self.dropout_2)(flatten)
        merged = concatenate([encoded_image, lstm_layer], axis=-1)

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
        # rgb_model = self.conv3d_lstm(channels=3, top_layer=False, stateful=False)
        # rgb_model = TimeDistributed(self.conv2d_lstm(channels=3, top_layer=False, stateful=False))
        # rgb_model = TimeDistributed(self.simonyan_4d(channels=3, top_layer=False, stateful=False))

        rgb_model = self.convolutional_LSTM(channels=3, top_layer=False)
        image_input = Input(shape=(None, self.input_shape[0], self.input_shape[1], 3))
        encoded_image = rgb_model(image_input)

        # of_model = self.conv3d_lstm(channels=3, top_layer=False, stateful=False)
        # of_model = TimeDistributed(self.conv2d_lstm(channels=3, top_layer=False, stateful=False))
        # of_model = TimeDistributed(self.simonyan_4d(channels=3, top_layer=False, stateful=False))

        of_model = self.convolutional_LSTM(channels=3, top_layer=False)
        of_input = Input(shape=(None, self.input_shape[0], self.input_shape[1], 3))
        encoded_of = of_model(of_input)
        
        # merged = add([encoded_image, encoded_of])
        merged = multiply([encoded_image, encoded_of])
        # merged = concatenate([encoded_image, encoded_of], axis=-1)
        merged = Dropout(.2)(merged)
        # dense = Dense(self.nb_dense_units, activation='relu')(merged)

        if self.nb_labels == 2:
            output = Dense(self.nb_labels, activation='sigmoid')(merged)
        else:
            output = Dense(self.nb_labels, activation='softmax')(merged)

        two_stream_model = Model(inputs=[image_input, of_input], outputs=[output])

        return two_stream_model

    def two_stream_stateful(self):
        rgb_model = TimeDistributed(self.conv2d_lstm_stateful(channels=3, top_layer=False))
        image_input = Input(shape=(self.seq_length, self.input_shape[0], self.input_shape[1], 3),
                            batch_shape=(self.batch_size, self.seq_length, self.input_shape[0], self.input_shape[1], 3))
        encoded_image = rgb_model(image_input)

        of_model = TimeDistributed(self.conv2d_lstm_stateful(channels=3, top_layer=False))
        of_input = Input(shape=(self.seq_length, self.input_shape[0], self.input_shape[1], 3),
                         batch_shape=(self.batch_size, self.seq_length, self.input_shape[0], self.input_shape[1], 3))
        encoded_of = of_model(of_input)

        merged = concatenate([encoded_image, encoded_of], axis=-1)
        print("Merged")
        if self.nb_labels == 2:
            output = Dense(self.nb_labels, activation='sigmoid')(merged)
        else:
            output = Dense(self.nb_labels, activation='softmax')(merged)

        two_stream_model = Model(inputs=[image_input, of_input], outputs=[output])
        return two_stream_model


    def conv3d_lstm(self, channels, top_layer=True, stateful=False):
        # Conv3d downsamples the time dimension. Which makes the coming dimensions weird.
        # Therefore, timedist conv.
        model = Sequential()
        model.add(Conv3D(filters=self.nb_conv_filters,
                         kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                         activation='relu'),
                         input_shape=(self.seq_length, self.input_shape[0], self.input_shape[1], 3),
                         batch_input_shape=(self.batch_size, self.seq_length,
                                            self.input_shape[0], self.input_shape[1], 3))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(BatchNormalization())
        model.add(Conv3D(filters=self.nb_conv_filters,
                         kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                         activation='relu'),
                         input_shape=(self.seq_length, self.input_shape[0], self.input_shape[1], 3),
                         batch_input_shape=(self.batch_size, self.seq_length,
                                            self.input_shape[0], self.input_shape[1], 3))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(BatchNormalization())
        # model.add(Conv3D(filters=self.nb_conv_filters,
        #                  kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
        #                  padding='same',
        #                  input_shape=(self.seq_length, self.input_shape[0], self.input_shape[1], channels),
        #                  batch_input_shape=(None, self.seq_length, self.input_shape[0], self.input_shape[1], channels),
        #                  activation='relu', kernel_initializer='he_uniform'))
        # # model.add(MaxPooling3D())
        # model.add(BatchNormalization())
        # model.add(Conv3D(filters=self.nb_conv_filters,
        #                  kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
        #                  padding='same',
        #                  input_shape=(self.seq_length, self.input_shape[0], self.input_shape[1], channels),
        #                  activation='relu', kernel_initializer='he_uniform'))
        # # model.add(MaxPooling3D())
        # model.add(BatchNormalization())
        model.add(TimeDistributed(Flatten()))
        model.add((LSTM(self.nb_lstm_units,
                        stateful=stateful,
                        dropout=self.dropout_2,
                        input_shape=(None, None, None),
                        return_sequences=True,
                        implementation=2)))
        model.add((LSTM(self.nb_lstm_units,
                        stateful=stateful,
                        dropout=self.dropout_2,
                        input_shape=(None, None, None),
                        return_sequences=True,
                        implementation=2)))
        model.add((LSTM(self.nb_lstm_units,
                        stateful=stateful,
                        dropout=self.dropout_2,
                        input_shape=(None, None, None),
                        return_sequences=True,
                        implementation=2)))
        model.add((LSTM(self.nb_lstm_units,
                        stateful=stateful,
                        dropout=self.dropout_2,
                        input_shape=(None, None, None),
                        return_sequences=True,
                        implementation=2)))
        if top_layer:
            if self.nb_labels == 2:
                print("2 labels, using sigmoid activation instead of softmax.")
                model.add(TimeDistributed(Dense(self.nb_labels, activation='sigmoid')))
                # model.add(Dense(self.nb_labels, activation='sigmoid'))
            else:
                model.add(TimeDistributed(Dense(self.nb_labels, activation='softmax')))
                # model.add(Dense(self.nb_labels, activation='softmax'))
        return model


#     def conv3d_lstm(self, channels, top_layer=True, stateful=False):
#         # Conv3d downsamples the time dimension. Which makes the coming dimensions weird.
#         # Therefore, timedist conv.
#         model = Sequential()
#         model.add(TimeDistributed(Convolution2D(filters=self.nb_conv_filters,
#                                                 kernel_size=(self.kernel_size,
#                                                              self.kernel_size),
#                                                 activation='relu'),
#                                   input_shape=(self.seq_length, self.input_shape[0], self.input_shape[1], 3),
#                                   batch_input_shape=(self.batch_size, self.seq_length,
#                                                      self.input_shape[0], self.input_shape[1], 3)))
#         model.add(TimeDistributed(MaxPooling2D()))
#         model.add(BatchNormalization())
#         model.add(TimeDistributed(Convolution2D(filters=self.nb_conv_filters,
#                                                 kernel_size=(self.kernel_size, self.kernel_size),
#                                                 activation='relu',
#                                                 kernel_initializer='he_uniform')))
#         model.add(TimeDistributed(MaxPooling2D()))
#         model.add(BatchNormalization())
#         # model.add(Conv3D(filters=self.nb_conv_filters,
#         #                  kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
#         #                  padding='same',
#         #                  input_shape=(self.seq_length, self.input_shape[0], self.input_shape[1], channels),
#         #                  batch_input_shape=(None, self.seq_length, self.input_shape[0], self.input_shape[1], channels),
#         #                  activation='relu', kernel_initializer='he_uniform'))
#         # # model.add(MaxPooling3D())
#         # model.add(BatchNormalization())
#         # model.add(Conv3D(filters=self.nb_conv_filters,
#         #                  kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
#         #                  padding='same',
#         #                  input_shape=(self.seq_length, self.input_shape[0], self.input_shape[1], channels),
#         #                  activation='relu', kernel_initializer='he_uniform'))
#         # # model.add(MaxPooling3D())
#         # model.add(BatchNormalization())
#         model.add(TimeDistributed(Flatten()))
#         model.add((LSTM(self.nb_lstm_units,
#                         stateful=stateful,
#                         dropout=self.dropout_2,
#                         input_shape=(None, None, None),
#                         return_sequences=True,
#                         implementation=2)))
#         model.add((LSTM(self.nb_lstm_units,
#                         stateful=stateful,
#                         dropout=self.dropout_2,
#                         input_shape=(None, None, None),
#                         return_sequences=True,
#                         implementation=2)))
#         model.add((LSTM(self.nb_lstm_units,
#                         stateful=stateful,
#                         dropout=self.dropout_2,
#                         input_shape=(None, None, None),
#                         return_sequences=True,
#                         implementation=2)))
#         model.add((LSTM(self.nb_lstm_units,
#                         stateful=stateful,
#                         dropout=self.dropout_2,
#                         input_shape=(None, None, None),
#                         return_sequences=True,
#                         implementation=2)))
#         if top_layer:
#             if self.nb_labels == 2:
#                 print("2 labels, using sigmoid activation instead of softmax.")
#                 model.add(TimeDistributed(Dense(self.nb_labels, activation='sigmoid')))
#                 # model.add(Dense(self.nb_labels, activation='sigmoid'))
#             else:
#                 model.add(TimeDistributed(Dense(self.nb_labels, activation='softmax')))
#                 # model.add(Dense(self.nb_labels, activation='softmax'))
#         return model

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
        # model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
        #                         activation='relu', kernel_initializer='he_uniform'))
        # model.add(MaxPooling2D())
        # model.add(BatchNormalization())
        # model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
        #                         activation='relu', kernel_initializer='he_uniform'))
        # model.add(MaxPooling2D())
        # model.add(BatchNormalization())
        # model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
        #                         activation='relu', kernel_initializer='he_uniform'))
        # model.add(MaxPooling2D())
        # model.add(BatchNormalization())
        # model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
        #                         activation='relu', kernel_initializer='he_uniform'))
        # model.add(BatchNormalization())
        # model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(3, 3),
        #                         activation='relu', kernel_initializer='he_uniform'))
        # model.add(BatchNormalization())
        model.add(TimeDistributed(Flatten()))
        if self.nb_lstm_layers >= 1:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=stateful,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            batch_input_shape=(self.batch_size, None, None, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers >= 2:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=stateful,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers >= 3:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=stateful,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers >= 4:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=stateful,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers >= 5:
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
        if self.nb_lstm_layers >= 2:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers >= 3:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers >= 4:
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
        # model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
        #                         activation='relu'))
        # model.add(MaxPooling2D())
        # model.add(BatchNormalization())
        # model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
        #                         activation='relu'))
        # model.add(MaxPooling2D())
        # model.add(BatchNormalization())
        # model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
        #                         activation='relu'))
        # model.add(MaxPooling2D())
        # model.add(BatchNormalization())
        # model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(self.kernel_size, self.kernel_size),
        #                         activation='relu'))
        # model.add(Convolution2D(filters=self.nb_conv_filters, kernel_size=(3, 3),
        #                         activation='relu'))
        model.add(TimeDistributed(Flatten()))
        if self.nb_lstm_layers == 4:
            print("4 lstm layers")
            model.add((LSTM(self.nb_lstm_units,
                            stateful=True,
                            dropout=self.dropout_2,
                            input_shape=(self.batch_size, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=True,
                            dropout=self.dropout_2,
                            input_shape=(self.batch_size, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=True,
                            dropout=self.dropout_2,
                            input_shape=(self.batch_size, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=True,
                            dropout=self.dropout_2,
                            input_shape=(self.batch_size, self.seq_length, None),
                            return_sequences=False,
                            implementation=2)))
        if self.nb_lstm_layers == 1:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=True,
                            dropout=self.dropout_2,
                            input_shape=(self.batch_size, self.seq_length, None),
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

    def inception_4d_input(self):
        model = Sequential()
        model.add(InceptionV3(include_top=False, input_shape=(self.input_shape[0],
                                                              self.input_shape[1],
                                                              3)))
        model.add(Flatten())
        if self.nb_labels == 2:
            print("2 labels, using sigmoid activation instead of softmax.")
            model.add(Dense(self.nb_labels, activation='sigmoid'))
        else:
            model.add(Dense(self.nb_labels, activation='softmax'))
        return model

    def inception_lstm_5d_input(self, top_layer=True):
        model = Sequential()
        model.add(TimeDistributed(InceptionV3(include_top=False,
                                              input_shape=(self.input_shape[0],
                                                          self.input_shape[1],
                                                          3)),
                                  input_shape=(self.seq_length,
                                               self.input_shape[0],
                                               self.input_shape[1],
                                               3)))
        model.add(TimeDistributed(Flatten()))
        if self.nb_lstm_layers == 1:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=False,
                            dropout=self.dropout_2,
                            input_shape=(None, self.seq_length, None),
                            return_sequences=True,
                            implementation=2)))
        model.add(Dropout(self.dropout_2))
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
        return model

    def convolutional_LSTM(self, channels=3, top_layer=True):
        model = Sequential()
        if self.nb_lstm_layers >= 1:
            model.add(ConvLSTM2D(filters=self.nb_lstm_units,
                                 kernel_size=(self.kernel_size, self.kernel_size),
                                 input_shape=(None, self.input_shape[0], self.input_shape[1], channels),
                                 padding='same', return_sequences=True))
            model.add(TimeDistributed(MaxPooling2D()))
            model.add(BatchNormalization())
        
        if self.nb_lstm_layers >= 2:
            model.add(ConvLSTM2D(filters=self.nb_lstm_units, kernel_size=(self.kernel_size, self.kernel_size),
                                 padding='same', return_sequences=True))
            model.add(TimeDistributed(MaxPooling2D()))
            model.add(BatchNormalization())

        if self.nb_lstm_layers >= 3:
            model.add(ConvLSTM2D(filters=self.nb_lstm_units, kernel_size=(self.kernel_size, self.kernel_size),
                                 padding='same', return_sequences=True))
            model.add(TimeDistributed(MaxPooling2D()))
            model.add(BatchNormalization())
        
        if self.nb_lstm_layers >= 4:
            model.add(ConvLSTM2D(filters=self.nb_lstm_units, kernel_size=(self.kernel_size, self.kernel_size),
                                 padding='same', return_sequences=True))
            model.add(TimeDistributed(MaxPooling2D()))
            model.add(BatchNormalization())

        if self.nb_lstm_layers >= 5:
            model.add(ConvLSTM2D(filters=self.nb_lstm_units, kernel_size=(self.kernel_size, self.kernel_size),
                                 padding='same', return_sequences=True))
            model.add(BatchNormalization())
        model.add(TimeDistributed(Flatten()))
        if top_layer:
            if self.nb_labels == 2:
                model.add((Dense(self.nb_labels, activation="sigmoid")))
            else:
                model.add((Dense(self.nb_labels, activation="softmax")))
        return model

    def conv2d_timedist_lstm_stateful(self, top_layer=True):
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
        model.add(TimeDistributed(Flatten()))
        if self.nb_lstm_layers == 4:
            model.add((LSTM(self.nb_lstm_units,
                            stateful=True,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=True,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=True,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            return_sequences=True,
                            implementation=2)))
            model.add((LSTM(self.nb_lstm_units,
                            stateful=True,
                            dropout=self.dropout_2,
                            input_shape=(None, None, None),
                            return_sequences=True,
                            implementation=2)))
            if top_layer:
                if self.nb_labels == 2:
                    print("2 labels, using sigmoid activation instead of softmax.")
                    model.add(TimeDistributed(Dense(self.nb_labels, activation='sigmoid')))
                else:
                    model.add(TimeDistributed(Dense(self.nb_labels, activation='softmax')))
            return model

