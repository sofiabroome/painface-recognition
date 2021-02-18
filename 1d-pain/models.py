import sys
sys.path.append('..')
import tensorflow as tf
import transformer


def get_transformer_model(config_dict):
    input_features = tf.keras.layers.Input(shape=(config_dict['video_pad_length'], config_dict['feature_dim']))
    target_sequence = tf.keras.layers.Input(shape=(config_dict['video_pad_length'], config_dict['nb_labels']))

    transformer_model = transformer.Transformer(config_dict)

    decoder_output = transformer_model(input_features, target_sequence)

    model = tf.keras.Model(inputs=[input_features, target_sequence], outputs=[decoder_output])
    model.summary()
    return model


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


def get_gru_attention_model(config_dict):
    input_features = tf.keras.layers.Input(shape=(config_dict['video_pad_length'], 1))

    x = tf.keras.layers.Masking(
        mask_value=0.,
        input_shape=(
            config_dict['video_pad_length'], config_dict['feature_dim'])
    )(input_features)

    for l_ind, l_units in enumerate(config_dict['layers']):
        # if l_ind == 0:
        #     x = input_features
        x, last_state = tf.keras.layers.GRU(
            l_units, return_sequences=True, return_state=True)(x)
    attention_layer = BahdanauAttention(10)
    # The attention_result is the context vector
    context_vectors = []
    for clip_ind in range(config_dict['video_pad_length']):
        attention_result, attention_weights = attention_layer(x[:,clip_ind,:], x)
        context_vectors.append(attention_result)
    x = tf.stack(context_vectors, axis=1)
    x = tf.keras.layers.GRU(8, return_sequences=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(2)(x)
    x = tf.keras.layers.Activation('softmax')(x)
    model = tf.keras.Model(inputs=[input_features], outputs=[x]) 
    model.summary()
    return model


def get_gru_model(config_dict):
    input_features = tf.keras.layers.Input(shape=(config_dict['video_pad_length'], 1))
    x = tf.keras.layers.Masking(
        mask_value=0., input_shape=(config_dict['video_pad_length'], 1))(input_features)

    # feature_enc1 = tf.keras.layers.GRU(32, return_sequences=True)
    for l_ind, l_units in enumerate(config_dict['layers']):
        # if l_ind == 0:
        #     x = input_features
        x = tf.keras.layers.GRU(l_units, return_sequences=True)(x)
    #feature_enc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2))
    # x = feature_enc2(input_features)
    # x = feature_enc2(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('softmax')(x)
    model = tf.keras.Model(inputs=[input_features], outputs=[x]) 
    model.summary()
    return model


def get_gru_return_state_model(config_dict):
    input_features = tf.keras.layers.Input(shape=(config_dict['video_pad_length'], 1))
    # feature_enc1 = tf.keras.layers.GRU(32, return_sequences=True)
    for l_ind, l_units in enumerate(config_dict['layers']):
        if l_ind == 0:
            x = input_features
        if l_ind + 1 == len(config_dict['layers']):
            x = tf.keras.layers.GRU(l_units, return_sequences=False)(x)
        else:
            x = tf.keras.layers.GRU(l_units, return_sequences=True)(x)
    #feature_enc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2))
    # x = feature_enc2(input_features)
    # x = feature_enc2(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(2)(x)
    x = tf.keras.layers.Activation('softmax')(x)
    model = tf.keras.Model(inputs=[input_features], outputs=[x]) 
    model.summary()
    return model


def get_dense_model(config_dict):
    input_features = tf.keras.layers.Input(shape=(config_dict['video_pad_length']))
    feature_enc1 = tf.keras.layers.Dense(2)
    feature_enc2 = tf.keras.layers.Dense(2)
    x = feature_enc1(input_features)
    x = feature_enc2(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('softmax')(x)
    model = tf.keras.Model(inputs=[input_features], outputs=[x])
    model.summary()
    return model


def get_identity_model(config_dict):
    input_features = tf.keras.layers.Input(shape=(config_dict['video_pad_length'], 1))
    feature_enc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config_dict['layers'][0]))
    x = feature_enc(input_features)
    # x = tf.keras.layers.Activation('softmax')(x)
    model = tf.keras.Model(inputs=[input_features], outputs=[x]) 
    model.summary()
    return model


