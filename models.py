from tensorflow.keras.layers import Convolution2D, MaxPooling2D, MaxPooling3D, GlobalAveragePooling2D, LSTM, Dense, Flatten, Conv2D
from tensorflow.keras.layers import Dropout, BatchNormalization, concatenate, add, average, Input, Conv3D, multiply, Activation
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential, Model


class MyModel:
    def __init__(self, config_dict):
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
        self.name = config_dict['model']
        self.input_shape = config_dict['input_height'], config_dict['input_width']
        self.nb_conv_filters = config_dict['nb_conv_filters']
        self.nb_lstm_units = config_dict['nb_lstm_units']
        self.kernel_size = config_dict['kernel_size']
        self.nb_labels = config_dict['nb_labels']
        self.dropout_2 = config_dict['dropout_2']
        self.dropout_1 = config_dict['dropout_1']
        self.seq_length = config_dict['seq_length']
        self.lr = config_dict['lr']
        self.optimizer = config_dict['optimizer']
        self.batch_size = config_dict['batch_size']
        self.nb_lstm_layers = config_dict['nb_lstm_layers']
        self.nb_dense_units = config_dict['nb_dense_units']

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

        if self.name == 'conv2d_informed':
            print("Conv2d informed model")
            self.model = self.conv2d_informed()

        if self.name == 'inception_lstm_5d_input':
            print('inception_lstm_5d_input')
            self.model = self.inception_lstm_5d_input()

        if self.name == 'inception_4d_input':
            print('inception_4d_input with imagenet weights')
            self.model, base_model = self.inception_4d_input(w='imagenet')
            # first: train only the top layers (which were randomly initialized)
            # i.e. freeze all convolutional InceptionV3 layers
            for layer in base_model.layers:
                layer.trainable = False
            self.base_model = base_model

        if self.name == 'inception_4d_input_from_scratch':
            print('inception_4d_input trained from scratch with random init')
            self.model, base_model = self.inception_4d_input(w=None)

        if self.name == '2stream':
            print('2stream')
            self.model = self.two_stream()

        if self.name == '2stream_5d_add':
            print('2stream 5D')
            self.model = self.two_stream_5d(fusion='add')

        if self.name == '2stream_5d_mult':
            print('2stream 5D')
            self.model = self.two_stream_5d(fusion='mult')

        if self.name == 'simonyan_2stream':
            print('Simonyan 2-stream with average fusion')
            self.model = self.simonyan_two_stream('average')

        if self.name == 'rodriguez_2stream':
            print('Rodriguez 2-stream with average fusion')
            self.model = self.two_stream_rodriguez('add')

        if self.name == '2stream_pretrained':
            print('2stream_pretrained')
            self.model = self.two_stream_pretrained()

        if self.name == 'convolutional_LSTM':
            print('Convolutional LSTM (not fully connected)')
            self.model = self.convolutional_LSTM(channels=3)

        if self.name == 'rodriguez':
            print('Rodriguez Deep pain model')
            self.model = self.rodriguez()

        if self.name == 'vgg16':
            print('VGG-16 trained from scratch, with 2 FC layers on top.')
            self.model = self.vgg16(w=None)

        if self.name == 'vgg16_GAP_dense':
            print('VGG-16 trained from scratch, then global avg pooling, then one FC layer.')
            self.model = self.vgg16_GAP_dense(w=None)


    def two_stream(self):
        # Functional API
        rgb_model, _ = self.inception_4d_input(w=None, top_layer=False)
        image_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))
        encoded_image = rgb_model(image_input)

        of_model, _ = self.inception_4d_input(w=None, top_layer=False)
        of_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))
        encoded_of = of_model(of_input)
        
        merged = add([encoded_image, encoded_of])

        merged = Dropout(.2)(merged)
        dense = Dense(self.nb_labels)(merged)

        if self.nb_labels == 2:
            output = Activation('sigmoid')(dense)
        else:
            output = Activation('softmax')(dense)

        two_stream_model = Model(inputs=[image_input, of_input], outputs=[output])
        return two_stream_model

    def two_stream_pretrained(self):
        # Functional API
        rgb_model, _ = self.inception_4d_input(w='imagenet', top_layer=False)
        image_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))
        encoded_image = rgb_model(image_input)

        of_model, _ = self.inception_4d_input(w='imagenet', top_layer=False)
        of_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))
        encoded_of = of_model(of_input)
        
        merged = add([encoded_image, encoded_of])

        merged = Dropout(.2)(merged)
        dense = Dense(self.nb_labels)(merged)

        if self.nb_labels == 2:
            output = Activation('sigmoid')(dense)
        else:
            output = Activation('softmax')(dense)

        two_stream_model = Model(inputs=[image_input, of_input], outputs=[output])
        return two_stream_model

    def rodriguez(self, top_layer=True):
        from tensorflow.keras.applications.vgg16 import VGG16
        image_input = Input(shape=(self.seq_length,
                                   self.input_shape[0],
                                   self.input_shape[1], 3))
        base_model = TimeDistributed(VGG16(weights='imagenet',
                                           include_top=False,  
                                           input_shape=(self.input_shape[0],
                                                        self.input_shape[1],
                                                        3)))(image_input)
        flatten = TimeDistributed(Flatten())(base_model)
        dense = Dense(self.nb_dense_units)(flatten)
        lstm_layer = LSTM(self.nb_lstm_units,
                          return_sequences=True)(dense)
        if top_layer:
            dense = Dense(self.nb_labels)(lstm_layer)
            if self.nb_labels == 2:
                output = Activation('sigmoid')(dense)
            else:
                output = Activation('softmax')(dense)
        else:
            output = lstm_layer
        model = Model(inputs=[image_input], outputs=[output])
        return model

    def vgg16(self, w):
        from tensorflow.keras.applications.vgg16 import VGG16
        image_input = Input(shape=(self.input_shape[0],
                                   self.input_shape[1], 3))
        base_model = VGG16(weights=w,
                           include_top=False,  
                           input_shape=(self.input_shape[0],
                                        self.input_shape[1],
                                        3))(image_input)
        flatten = Flatten()(base_model)
        dense_1 = Dense(4096)(flatten)                 # According to the paper, the two 
        dense_2 = Dense(self.nb_dense_units)(dense_1)  # FC-layers should have 4096 units each.
        if self.nb_labels == 2:
            dense_3 = Dense(self.nb_labels)(dense_2)
            output = Activation('sigmoid')(dense_3)
        model = Model(inputs=[image_input], outputs=[output])
        return model

    def vgg16_GAP_dense(self, w):
        from tensorflow.keras.applications.vgg16 import VGG16
        image_input = Input(shape=(self.input_shape[0],
                                   self.input_shape[1], 3))
        base_model = VGG16(weights=w,
                           include_top=False,  
                           input_shape=(self.input_shape[0],
                                        self.input_shape[1],
                                        3))(image_input)
        x = GlobalAveragePooling2D()(base_model)
        dense_1 = Dense(self.nb_dense_units, activation='relu')(x)
        if self.nb_labels == 2:
            dense_2 = Dense(self.nb_labels)(dense_1)
            output = Activation('sigmoid')(dense_2)
        model = Model(inputs=[image_input], outputs=[output])
        return model

    def two_stream_5d(self, fusion):

        rgb_model = self.convolutional_LSTM(channels=3, top_layer=False)
        input_array = Input(shape=(None, None, self.input_shape[0], self.input_shape[1], 3))
        encoded_image = rgb_model(input_array[0,:])

        of_model = self.convolutional_LSTM(channels=3, top_layer=False)
        encoded_of = of_model(input_array[1,:])

        if fusion == 'add':
            merged = add([encoded_image, encoded_of])
        if fusion == 'mult':
            merged = multiply([encoded_image, encoded_of])
        if fusion == 'concat':
            merged = concatenate([encoded_image, encoded_of], axis=-1)

        merged = Dropout(.2)(merged)
        dense = Dense(self.nb_labels)(merged)

        if self.nb_labels == 2:
            output = Activation('sigmoid')(dense)
        else:
            output = Activation('softmax')(dense)

        two_stream_model = Model(inputs=[input_array], outputs=[output])

        return two_stream_model

    def two_stream_rodriguez(self, fusion):
        rgb_model = self.rodriguez(top_layer=False)
        image_input = Input(shape=(None, self.input_shape[0], self.input_shape[1], 3))
        encoded_image = rgb_model(image_input)

        of_model = self.rodriguez(top_layer=False)
        of_input = Input(shape=(None, self.input_shape[0], self.input_shape[1], 3))
        encoded_of = of_model(of_input)

        if fusion == 'add':
            merged = add([encoded_image, encoded_of])
        if fusion == 'mult':
            merged = multiply([encoded_image, encoded_of])
        if fusion == 'concat':
            merged = concatenate([encoded_image, encoded_of], axis=-1)

        merged = Dropout(.2)(merged)
        dense = Dense(self.nb_labels)(merged)

        if self.nb_labels == 2:
            output = Activation('sigmoid')(dense)
        else:
            output = Activation('softmax')(dense)

        two_stream_model = Model(inputs=[image_input, of_input], outputs=[output])

        return two_stream_model

    def simonyan_spatial_stream(self):
        model = Sequential()
        model.add(Conv2D(filters=96, kernel_size=(7,7), strides=(2,2), activation='relu',
                  input_shape=(self.input_shape[0], self.input_shape[1], 3),
                  batch_input_shape=(self.batch_size,
                                     self.input_shape[0],
                                     self.input_shape[1],
                                     3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), activation='relu'))
        model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu',
                  input_shape=(self.input_shape[0], self.input_shape[1], 3)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu',
                  input_shape=(self.input_shape[0], self.input_shape[1], 3)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu',
                  input_shape=(self.input_shape[0], self.input_shape[1], 3)))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Dense(2048))
        model.add(Dense(self.nb_labels))
        if self.nb_labels == 2:
            print("2 labels, using sigmoid activation instead of softmax.")
            model.add(Activation('sigmoid'))
        else:
            model.add(Activation('softmax'))
        print(model.summary())
        return model

    def simonyan_temporal_stream(self):
        model = Sequential()
        model.add(Conv2D(filters=96, kernel_size=(7,7), strides=(2,2), data_format='channels_first', activation='relu',
                  input_shape=(2*self.seq_length, self.input_shape[0], self.input_shape[1]),
                  batch_input_shape=(self.batch_size, 2*self.seq_length,
                                     self.input_shape[0], self.input_shape[1])))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), activation='relu',
                  input_shape=(self.input_shape[0], self.input_shape[1], 3), data_format='channels_first'))
        # model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu',
                  input_shape=(self.input_shape[0], self.input_shape[1], 3), data_format='channels_first'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu',
                  input_shape=(self.input_shape[0], self.input_shape[1], 3), data_format='channels_first'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu',
                  input_shape=(self.input_shape[0], self.input_shape[1], 3), data_format='channels_first'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Dense(2048))
        model.add(Dense(self.nb_labels))
        if self.nb_labels == 2:
            print("2 labels, using sigmoid activation instead of softmax.")
            model.add(Activation('sigmoid'))
        else:
            model.add(Activation('softmax'))
        print(model.summary())
        return model

    def simonyan_two_stream(self, fusion):
        
        rgb_model = self.simonyan_spatial_stream()
        image_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))
        rgb_class_scores = rgb_model(image_input)

        of_model = self.simonyan_temporal_stream()
        of_input = Input(shape=(2*self.seq_length, self.input_shape[0], self.input_shape[1]))
        flow_class_scores = of_model(of_input)

        if fusion == 'average':
            output = average([rgb_class_scores, flow_class_scores])
        if fusion == 'svm':
            print('SVM implementation TODO.')

        two_stream_model = Model(inputs=[image_input, of_input], outputs=[output])

        return two_stream_model

    def inception_4d_input(self, w, top_layer=True):
        image_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))
        base_model = InceptionV3(weights=w, include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        if top_layer:
            dense = Dense(self.nb_labels)(x)
            if self.nb_labels == 2:
                predictions = Activation('sigmoid')(dense)
            else:
                predictions = Activation('softmax')(dense)
        else:
            predictions = x
        model = Model(inputs=base_model.input, outputs=predictions)
        return model, base_model

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
            model.add(Dense(self.nb_labels))
            if self.nb_labels == 2:
                print("2 labels, using sigmoid activation instead of softmax.")
                model.add(Activation('sigmoid'))
            else:
                model.add(Activation('softmax'))
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
                model.add(Dense(self.nb_labels))
                model.add(Activation('sigmoid'))
            else:
                model.add(Dense(self.nb_labels))
                model.add(Activation('softmax'))
        return model

