from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, LSTM, Dense, Flatten, Conv2D
from tensorflow.keras.layers import Dropout, BatchNormalization, Input, Activation
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
import tensorflow as tf
import i3d


class MyModel(tf.keras.Model):
    def __init__(self, config_dict):
        super(MyModel, self).__init__()
        """
        A class to build the preferred model.
        :param config_dict: dict
        """
        self.model_name = config_dict['model']
        self.config_dict = config_dict
        self.height = config_dict['input_height']
        self.width = config_dict['input_width']
        self.nb_lstm_units = config_dict['nb_lstm_units']
        self.nb_labels = config_dict['nb_labels']
        self.dropout_2 = config_dict['dropout_2']
        self.dropout_1 = config_dict['dropout_1']
        self.seq_length = config_dict['seq_length']
        self.lr = config_dict['lr']
        self.optimizer = config_dict['optimizer']
        self.batch_size = config_dict['batch_size']
        self.nb_lstm_layers = config_dict['nb_lstm_layers']
        
        if not self.config_dict['save_features']:
            if self.config_dict['video_level_mode']:
                training = True if config_dict['train_video_level_features'] else False
                self.video_features_model = self.config_dict['video_features_model']
                config_dict['model'] = self.video_features_model
                self.model_name = 'only_train_video_feats'
                if self.video_features_model == 'video_level_network':
                    self.model = self.video_level_network()
                if self.video_features_model == 'video_level_preds_attn_network':
                    self.model = self.video_level_preds_attn_network(training=training)
                if self.video_features_model == 'video_level_preds_attn_gru_network':
                    self.model = self.video_level_preds_attn_gru_network(training=training)
                if self.video_features_model == 'video_level_mil_feats':
                    self.model = self.video_level_mil_feats(training=training)
                if self.video_features_model == 'video_level_mil_feats_preds':
                    self.model = self.video_level_mil_feats_preds()
                if self.video_features_model == 'video_fc_model':
                    self.model = self.video_fc_model()
                if self.video_features_model == 'video_conv_seq_model':
                    self.model = self.video_conv_seq_model()

        if self.model_name == 'i3d_2stream':
            print('I3D') 
            self.model = self.i3d_2stream(fusion='add')

        if self.model_name == 'conv2d_timedist_lstm':
            print("Conv2d-lstm model timedist")
            self.model = self.conv2d_timedist_lstm()

        if self.model_name == 'conv2d_timedist_lstm_stateful':
            print("Stateful timedist conv2d-lstm model")
            self.model = self.conv2d_timedist_lstm_stateful()

        if self.model_name == 'conv2d_lstm':
            print("Conv2d-lstm model")
            self.model = self.conv2d_lstm(channels=3)

        if self.model_name == 'conv3d_lstm':
            print("Conv3d-lstm model")
            self.model = self.conv3d_lstm(channels=3)

        if self.model_name == 'conv2d_lstm_5d':
            print("Conv2d-lstm model 5D")
            self.model = self.conv2d_lstm_5d(channels=3)

        if self.model_name == 'conv2d_lstm_stateful':
            print("Conv2d-lstm model stateful")
            self.model = self.conv2d_lstm_stateful()

        if self.model_name == 'conv2d_lstm_informed':
            print("Conv2d-lstm model informed")
            self.model = self.conv2d_lstm_informed()

        if self.model_name == 'conv2d_informed':
            print("Conv2d informed model")
            self.model = self.conv2d_informed()

        if self.model_name == 'inception_lstm_5d_input':
            print('inception_lstm_5d_input')
            self.model = self.inception_lstm_5d_input()

        if self.model_name == 'inception_4d_input':
            print('inception_4d_input with imagenet weights')
            self.model, base_model = self.inception_4d_input(w='imagenet')
            # first: train only the top layers (which were randomly initialized)
            # i.e. freeze all convolutional InceptionV3 layers
            for layer in base_model.layers:
                layer.trainable = False
            self.base_model = base_model

        if self.model_name == 'inception_4d_input_from_scratch':
            print('inception_4d_input trained from scratch with random init')
            self.model, base_model = self.inception_4d_input(w=None)

        if self.model_name == '2stream':
            print('2stream')
            self.model = self.two_stream()

        if self.model_name == '2stream_5d_add':
            print('2stream 5D')
            if self.config_dict['inference_only']:
                self.model = self.two_stream_5d(fusion='add', train=False)
            else:
                self.model = self.two_stream_5d(fusion='add')

        if self.model_name == '2stream_5d_mult':
            print('2stream 5D')
            self.model = self.two_stream_5d(fusion='mult')

        if self.model_name == 'simonyan_2stream':
            print('Simonyan 2-stream with average fusion')
            self.model = self.simonyan_two_stream('average')

        if self.model_name == 'rodriguez_2stream':
            print('Rodriguez 2-stream with average fusion')
            self.model = self.two_stream_rodriguez('add')

        if self.model_name == '2stream_pretrained':
            print('2stream_pretrained')
            self.model = self.two_stream_pretrained()

        if self.model_name == 'convolutional_LSTM':
            print('Convolutional LSTM (not fully connected)')
            self.model = self.convolutional_LSTM(channels=3)

        if self.model_name == 'clstm_functional':
            print('C-LSTM, functional')
            self.model = self.clstm()

        if self.model_name == 'rodriguez':
            print('Rodriguez Deep pain model')
            self.model = self.rodriguez()

        if self.model_name == 'vgg16':
            print('VGG-16 trained from scratch, with 2 FC layers on top.')
            self.model = self.vgg16(w=None)

        if self.model_name == 'vgg16_GAP_dense':
            print('VGG-16 trained from scratch, then global avg pooling, then one FC layer.')
            self.model = self.vgg16_GAP_dense(w=None)

    def i3d_classification_block(self, x, dropout_prob, classes, name_str):
        # Classification block
        print('\nCLASSIFICATION HEAD SHAPES: ')
        print(x.shape)
        # x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)
        x = AveragePooling3D((2, 4, 4), strides=(1, 1, 1), padding='valid',
            name='global_avg_pool_{}'.format(name_str))(x)
        print(x.shape)
        x = Dropout(dropout_prob)(x)
        x = i3d.conv3d_bn(x, classes, 1, 1, 1, padding='same', 
            use_bias=True, use_activation_fn=False, use_bn=False,
            name='Conv3d_6a_1x1_{}'.format(name_str))
        print(x.shape)
 
        num_frames_remaining = int(x.shape[1])
        print('num_frames_remaining :', num_frames_remaining)
        x = Reshape((num_frames_remaining, classes))(x)

        # logits (raw scores for each class)
        logits = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)
        return logits

    def i3d_2stream(self, fusion, train=True):

        rgb_model = i3d.Inception_Inflated3d(
            include_top=False,
            weights='rgb_kinetics_only',
            input_shape=(self.config_dict['seq_length'], self.height, self.width, 3),
            classes=self.nb_labels)
        input_array = Input(shape=(None, self.config_dict['seq_length'], self.height, self.width, 3))
        encoded_image = rgb_model(input_array[:, 0, :])
        rgb_logits = self.i3d_classification_block(encoded_image,
            dropout_prob=self.dropout_2, classes=self.nb_labels, name_str='rgb')

        flow_model = i3d.Inception_Inflated3d(
            include_top=False,
            weights='flow_kinetics_only',
            input_shape=(self.config_dict['seq_length'], self.height, self.width, 2),
            classes=self.nb_labels)
        encoded_flow = flow_model(input_array[:, 1, :, :, :, :2])
        flow_logits = self.i3d_classification_block(encoded_flow,
            dropout_prob=self.dropout_2, classes=self.nb_labels, name_str='flow')

        features = encoded_image + encoded_flow

        logits = rgb_logits + flow_logits  # How it's done in the deepmind repo.

        output = Activation('softmax')(logits)

        if self.config_dict['return_last_clstm']:
            outputs = [output, features]
        else:
            outputs = [output]

        model = tf.keras.Model(inputs=[input_array], outputs=outputs)
        return model
    
    def two_stream_5d(self, fusion, train=True):

        rgb_model = self.convolutional_LSTM(channels=3, top_layer=False)
        # rgb_model = self.clstm(channels=3, top_layer=False, bn=True)
        input_array = Input(shape=(None, self.config_dict['seq_length'],
                            self.height, self.width, 3))
        encoded_image = rgb_model(input_array[:, 0, :])
        # encoded_image = rgb_model(input_array[0, :])

        flow_model = self.convolutional_LSTM(channels=3, top_layer=False)
        # flow_model = self.clstm(channels=3, top_layer=False, bn=True)
        encoded_flow = flow_model(input_array[:, 1, :])
        # encoded_flow = flow_model(input_array[1, :])

        if fusion == 'add':
            merged = tf.keras.layers.add([encoded_image, encoded_flow])
        if fusion == 'mult':
            merged = tf.keras.layers.multiply([encoded_image, encoded_flow])
        if fusion == 'concat':
            merged = tf.keras.layers.concatenate([encoded_image, encoded_flow], axis=-1)

        merged_flat = Flatten()(merged)

        if train:
            merged_flat = Dropout(self.dropout_1)(merged_flat)
        dense = Dense(self.nb_labels)(merged_flat)

        if self.nb_labels == 2:
            # output = Activation('sigmoid')(dense)
            output = Activation('softmax')(dense)
        else:
            output = Activation('softmax')(dense)

        if self.config_dict['return_last_clstm']:
            outputs = [output, merged]
        else:
            outputs = [output]

        two_stream_model = tf.keras.Model(inputs=[input_array], outputs=outputs)

        return two_stream_model

    def convolutional_LSTM(self, channels=3, top_layer=True):
        model = Sequential()
        if self.nb_lstm_layers >= 1:
            model.add(ConvLSTM2D(filters=self.nb_lstm_units,
                                 kernel_size=(self.config_dict['kernel_size'], self.config_dict['kernel_size']),
                                 input_shape=(self.config_dict['seq_length'], self.height, self.width, channels),
                                 padding='same', return_sequences=True))
            model.add(TimeDistributed(MaxPooling2D()))
            model.add(BatchNormalization())
        
        if self.nb_lstm_layers >= 2:
            model.add(ConvLSTM2D(filters=self.nb_lstm_units, kernel_size=(self.config_dict['kernel_size'], self.config_dict['kernel_size']),
                                 padding='same', return_sequences=True))
            model.add(TimeDistributed(MaxPooling2D()))
            model.add(BatchNormalization())

        if self.nb_lstm_layers >= 3:
            model.add(ConvLSTM2D(filters=self.nb_lstm_units, kernel_size=(self.config_dict['kernel_size'], self.config_dict['kernel_size']),
                                 padding='same', return_sequences=True))
            model.add(TimeDistributed(MaxPooling2D()))
            model.add(BatchNormalization())
        
        if self.nb_lstm_layers >= 4:
            model.add(ConvLSTM2D(filters=self.nb_lstm_units, kernel_size=(self.config_dict['kernel_size'], self.config_dict['kernel_size']),
                                 padding='same', return_sequences=True))
            model.add(TimeDistributed(MaxPooling2D()))
            model.add(BatchNormalization())

        if self.nb_lstm_layers >= 5:
            model.add(ConvLSTM2D(filters=self.nb_lstm_units, kernel_size=(self.config_dict['kernel_size'], self.config_dict['kernel_size']),
                                 padding='same', return_sequences=True))
            model.add(BatchNormalization())

        if top_layer:
            model.add(Flatten())
            model.add(Dense(self.nb_labels))
            if self.nb_labels == 2:
                model.add(Activation('sigmoid'))
            else:
                model.add(Activation('softmax'))

        return model

    def clstm_block(self, input_tensor, nb_hidden, ks1, ks2, pooling,
                    batch_normalization, return_sequences):
        """
        x: input tensor
        nb_hidden: int
        ks: int
        pooling: str 'max'|'avg'
        batch_normalization: bool
        return_sequences: bool
        """
        # Kernel regularizer
        if self.config_dict['kernel_regularizer'] is None:
            reg = None
        else:
            reg = tf.keras.regularizers.l2(self.config_dict['kernel_regularizer'])
        # ConvLSTM2D layer
        clstm_output = tf.keras.layers.ConvLSTM2D(
            filters=nb_hidden,
            kernel_size=(ks1, ks2),
            padding=self.config_dict['padding_clstm'],
            strides=self.config_dict['strides_clstm'],
            kernel_regularizer=reg,
            dropout=self.config_dict['dropout_clstm'],
            return_sequences=return_sequences)(input_tensor)
        if return_sequences:
            # Maxpooling layer per time slice
            if pooling == 'max':
                x = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.MaxPooling2D())(
                    inputs=clstm_output)
            if pooling == 'avg':
                x = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.AveragePooling2D())(
                    inputs=clstm_output)
        else:
            if pooling == 'max':
                x = tf.keras.layers.MaxPooling2D()(inputs=clstm_output)
            if pooling == 'avg':
                x = tf.keras.layers.AveragePooling2D()(inputs=clstm_output)
        print(x)
        # Normalize according to batch statistics
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
            print(x)
        return x, clstm_output

    def clstm(self, channels=3, top_layer=True, bn=True):
        """x: 5D tensor, sequence of images
           bn: bool, whether to batch normalize
           return: x, the transformed input sequence."""

        input_layer = Input(shape=(
            self.config_dict['seq_length'],
            self.height, self.width, channels))
        layers = self.config_dict['nb_lstm_layers'] * [self.config_dict['nb_lstm_units']]
        rs = self.config_dict['return_sequences']
        nb_clstm_layers = len(layers)

        print(input)
        x = input_layer
        for l in range(nb_clstm_layers):
            name_scope = 'block' + str(l + 1)
            with tf.name_scope(name_scope):
                x, clstm_output = self.clstm_block(
                    x, nb_hidden=layers[l],
                    ks1=self.config_dict['kernel_size'],
                    ks2=self.config_dict['kernel_size'],
                    pooling=self.config_dict['pooling_method'],
                    batch_normalization=bn,
                    return_sequences=rs[l])
                print('x: ', x)
                print('clstm_output: ', clstm_output)

        if top_layer:
            with tf.name_scope('fully_con'):
                if self.config_dict['only_last_element_for_fc'] == 'yes':
                    # Only pass on the last element of the sequence to FC.
                    # return_seq is True just to save it in the graph for gradcam.
                    x = tf.keras.layers.Flatten()(x[:, -1, :, :, :])
                else:
                    x = tf.keras.layers.Flatten()(x)
                print(x)
                x = tf.keras.layers.Dense(units=self.config_dict['nb_labels'])(x)
                print(x)

        if self.config_dict['return_last_clstm']:
            model = tf.keras.Model(inputs=[input_layer], outputs=[x, clstm_output])
        else:
            model = tf.keras.Model(inputs=[input_layer], outputs=[x])

        return model

    def video_level_network(self):
        input_features = Input(shape=(None, self.config_dict['feature_dim']))
        input_preds = Input(shape=(None, 2))
        gru = tf.keras.layers.GRU(
            self.config_dict['nb_units_1'], return_sequences=True)
        gru_2 = tf.keras.layers.GRU(
            self.config_dict['nb_labels'], return_sequences=False)
        x = gru(input_features)
        x = gru_2(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=self.config_dict['nb_labels'])(x)
        x = Activation('softmax')(x)

        model = tf.keras.Model(inputs=[input_features, input_preds], outputs=[x])
        model.summary()

        return model

    def video_fc_model(self):
        input_features = Input(shape=(self.config_dict['video_pad_length'], self.config_dict['feature_dim']))
        input_preds = Input(shape=(self.config_dict['video_pad_length'], 2))
        x_preds = tf.keras.layers.Flatten()(input_preds)
        # x_feats = tf.keras.layers.Flatten()(input_features)
        # x = tf.keras.layers.concatenate([x_preds, x_feats], axis=1)
        # x = tf.keras.layers.GlobalAveragePooling1D()(input_features)

        x = tf.keras.layers.Dense(units=self.config_dict['nb_units_2'])(x_preds)
        x = tf.keras.layers.Dense(units=self.config_dict['nb_labels'])(x)
        x = Activation('softmax')(x)

        model = tf.keras.Model(inputs=[input_features, input_preds], outputs=[x])
        model.summary()

        return model

    def video_conv_seq_model(self):
        # input_features = Input(shape=(self.config_dict['video_pad_length'], self.config_dict['feature_dim']))
        # input_preds = Input(shape=(self.config_dict['video_pad_length'], 2))
        input_features = Input(shape=(None, self.config_dict['feature_dim']))
        input_preds = Input(shape=(None, 2))

        enc1_feats = tf.keras.layers.Conv1D(filters=self.config_dict['nb_units_1'],
                                            kernel_size=self.config_dict['kernel_size'],
                                            padding='same',
                                            activation='relu')
        enc2_feats = tf.keras.layers.Conv1D(filters=self.config_dict['nb_labels'],
                                            kernel_size=self.config_dict['kernel_size'],
                                            padding='same',
                                            activation='relu')
        enc3_rnn = tf.keras.layers.GRU(self.config_dict['nb_labels'], return_sequences=True)
        # enc2_feats = tf.keras.layers.Conv1D(self.config_dict['nb_labels'])
        x_feats = enc1_feats(input_features)
        x_feats = tf.keras.layers.BatchNormalization()(x_feats)
        x_feats = Dropout(self.config_dict['dropout_2'])(x_feats)
        x_feats = enc2_feats(x_feats)
        # x_feats = tf.keras.layers.Activation('relu')(x_feats)
        # x_feats = tf.keras.layers.multiply([x_feats, input_preds])
        # x_feats = enc3_rnn(x_feats)
        # x_feats = enc2_feats(x_feats)
        # x_feats = tf.keras.layers.Flatten()(x_feats)
        # x_feats = tf.keras.layers.Flatten()(input_features)
        # x = tf.keras.layers.concatenate([x_preds, x_feats], axis=1)
        # x = tf.keras.layers.GlobalAveragePooling1D()(input_features)

        # x = tf.keras.layers.Dense(units=self.config_dict['nb_units_1'])(x_feats)
        # x = tf.keras.layers.Dense(units=self.config_dict['nb_labels'])(x_feats)
        # x = Activation('softmax')(x)

        model = tf.keras.Model(inputs=[input_features, input_preds], outputs=[x_feats])
        model.summary()

        return model

    def video_level_mil_feats_old(self):

        input_features = Input(shape=(None, self.config_dict['feature_dim']))
        input_preds = Input(shape=(None, 2))
        # input_features = Input(shape=(self.config_dict['video_pad_length'], self.config_dict['feature_dim']))
        # input_preds = Input(shape=(self.config_dict['video_pad_length'], 2))

        feature_enc1 = tf.keras.layers.GRU(
            self.config_dict['nb_units_1'], return_sequences=True)
        feature_enc2 = tf.keras.layers.GRU(
            self.config_dict['nb_labels'], return_sequences=True)

        x_feats = feature_enc1(input_features)
        x_feats = feature_enc2(x_feats)
        preds_seq_from_feats = Activation('softmax')(x_feats)
        # preds_seq_from_feats = x_feats

        model = tf.keras.Model(inputs=[input_features, input_preds], outputs=[preds_seq_from_feats])
        model.summary()

        return model

    def video_level_mil_feats_preds(self):

        input_features = Input(shape=(self.config_dict['video_pad_length'], self.config_dict['feature_dim']))
        input_preds = Input(shape=(self.config_dict['video_pad_length'], 2))
        # Preds module
        x_preds = tf.keras.layers.Flatten()(input_preds)
        x_preds = tf.keras.layers.Dense(units=self.config_dict['nb_units_1'])(x_preds)
        x_preds = tf.keras.layers.Dense(units=self.config_dict['nb_labels'])(x_preds)
        # preds_one_from_preds = Activation('softmax')(x_preds)
        preds_one_from_preds = x_preds

        # Features module
        feature_enc1 = tf.keras.layers.GRU(
            self.config_dict['nb_units_1'], return_sequences=True)
        feature_enc2 = tf.keras.layers.GRU(
            self.config_dict['nb_labels'], return_sequences=True)

        x_feats = feature_enc1(input_features)
        x_feats = feature_enc2(x_feats)
        # preds_seq_from_feats = Activation('softmax')(x_feats)
        preds_seq_from_feats = x_feats

        model = tf.keras.Model(inputs=[input_features, input_preds], outputs=[preds_seq_from_feats, preds_one_from_preds])
        model.summary()

        return model

    def video_level_preds_attn_gru_network(self, training):

        input_features = Input(shape=(self.config_dict['video_pad_length'], self.config_dict['feature_dim']))
        input_preds = Input(shape=(self.config_dict['video_pad_length'], 2))

        reg = tf.keras.regularizers.l2(self.config_dict['l2_weight'])
        
        # FEATURES
        # feature_enc1 = tf.keras.layers.GRU(
        #     self.config_dict['nb_units_1'],
        #     kernel_regularizer=reg,
        #     recurrent_regularizer=reg,
        #     return_sequences=True)
        feature_enc1 = tf.keras.layers.GRU(
            self.config_dict['nb_units_1'],
            return_sequences=True)
        # feature_enc11 = tf.keras.layers.GRU(
        #     self.config_dict['nb_units_2'], return_sequences=True)
        feature_enc2 = tf.keras.layers.GRU(
            self.config_dict['nb_labels'], return_sequences=True)

        x = feature_enc1(input_features)
        if training:
            x = Dropout(self.config_dict['dropout_2'])(x)
        # x = feature_enc11(x)
        x = feature_enc2(x)
        
        # PREDS
        preds_enc = tf.keras.layers.GRU(
            self.config_dict['nb_labels'], return_sequences=True)

        preds = preds_enc(input_preds)
        
        if self.config_dict['merge_attn'] == 'mult':
            x = tf.keras.layers.multiply([x, preds])
        if self.config_dict['merge_attn'] == 'add':
            x = tf.keras.layers.add([x, preds])

        x = tf.keras.layers.BatchNormalization()(x, training=training)
        # x = tf.keras.layers.GlobalMaxPooling1D()(x)
        # x = tf.keras.layers.GlobalAveragePooling1D()(x)
        # x = tf.keras.layers.Dense(units=self.config_dict['nb_labels'])(x)
        x = Activation('softmax')(x)

        model = tf.keras.Model(inputs=[input_features, input_preds], outputs=[x])
        model.summary()

        return model

    def video_level_mil_feats(self, training):

        input_features = Input(shape=(self.config_dict['video_pad_length'], self.config_dict['feature_dim']))
        input_preds = Input(shape=(self.config_dict['video_pad_length'], 2))

        # FEATURES
        feature_enc1 = tf.keras.layers.GRU(
            self.config_dict['nb_units_1'],
            return_sequences=True)
        feature_enc2 = tf.keras.layers.GRU(
            self.config_dict['nb_labels'], return_sequences=True)

        x = feature_enc1(input_features)
        if training:
            x = Dropout(self.config_dict['dropout_2'])(x)
        x = feature_enc2(x)
        
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = Activation('softmax')(x)

        model = tf.keras.Model(inputs=[input_features, input_preds], outputs=[x])
        model.summary()

        return model

    def video_level_preds_attn_network(self, training):

        input_features = Input(shape=(self.config_dict['video_pad_length'], self.config_dict['feature_dim']))
        input_preds = Input(shape=(self.config_dict['video_pad_length'], 2))

        reg = tf.keras.regularizers.l2(self.config_dict['l2_weight'])
        
        # FEATURES
        fc_timedist_1 = TimeDistributed(tf.keras.layers.Dense(self.config_dict['nb_units_1']))
        fc_timedist_2 = TimeDistributed(tf.keras.layers.Dense(self.config_dict['nb_labels']))

        x = fc_timedist_1(input_features)
        x = fc_timedist_2(x)
        if training:
            x = Dropout(self.config_dict['dropout_2'])(x)

        
        # PREDS
        preds_enc_1 = TimeDistributed(tf.keras.layers.Dense(self.config_dict['nb_labels']))

        preds = preds_enc_1(input_preds)

        x = tf.keras.layers.multiply([x, preds])

        x = tf.keras.layers.BatchNormalization()(x, training=training)
        # x = tf.keras.layers.GlobalMaxPooling1D()(x)
        # x = tf.keras.layers.GlobalAveragePooling1D()(x)
        # x = tf.keras.layers.Dense(units=self.config_dict['nb_labels'])(x)
        x = Activation('softmax')(x)

        model = tf.keras.Model(inputs=[input_features, input_preds], outputs=[x])
        model.summary()

        return model
