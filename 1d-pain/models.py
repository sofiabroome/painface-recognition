import tensorflow as tf


def get_gru_model(config_dict):
    input_features = tf.keras.layers.Input(shape=(config_dict['T'], 1))
    # feature_enc1 = tf.keras.layers.GRU(32, return_sequences=True)
    for l_ind, l_units in enumerate(config_dict['layers']):
        if l_ind == 0:
            x = input_features
        x = tf.keras.layers.GRU(l_units, return_sequences=True)(x)
    #feature_enc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2))
    # x = feature_enc2(input_features)
    # x = feature_enc2(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('softmax')(x)
    model = tf.keras.Model(inputs=[input_features], outputs=[x]) 
    model.summary()
    return model


def get_dense_model(T):
    input_features = tf.keras.layers.Input(shape=(T))
    feature_enc1 = tf.keras.layers.Dense(32)
    feature_enc2 = tf.keras.layers.Dense(2)
    #feature_enc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2))
    x = feature_enc1(input_features)
    x = feature_enc2(x)
    model = tf.keras.Model(inputs=[input_features], outputs=[x]) 
    model.summary()
    return model


def get_identity_model(T):
    input_features = tf.keras.layers.Input(shape=(T, 1))
    feature_enc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2))
    #feature_enc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2))
    x = feature_enc(input_features)
    x = tf.keras.layers.Activation('softmax')(x)
    model = tf.keras.Model(inputs=[input_features], outputs=[x]) 
    model.summary()
    return model

