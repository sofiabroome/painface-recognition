import tensorflow as tf

# Encoder, decoder and mh-attention classes from
# https://trungtran.io/2019/04/29/create-the-transformer-with-tensorflow-2-0/


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.query_size = model_size // h
        self.key_size = model_size // h
        self.value_size = model_size // h
        self.h = h
        self.wq = [tf.keras.layers.Dense(self.query_size) for _ in range(h)]
        self.wk = [tf.keras.layers.Dense(self.key_size) for _ in range(h)]
        self.wv = [tf.keras.layers.Dense(self.value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, query, value):
        # query has shape (batch, query_len, model_size)
        # value has shape (batch, value_len, model_size)
        heads = []
        for i in range(self.h):
            score = tf.matmul(self.wq[i](query), self.wk[i](value), transpose_b=True)

            # Here we scale the score as described in the paper
            score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
            # score has shape (batch, query_len, value_len)

            alignment = tf.nn.softmax(score, axis=2)
            # alignment has shape (batch, query_len, value_len)

            head = tf.matmul(alignment, self.wv[i](value))
            # head has shape (batch, decoder_len, value_size)
            heads.append(head)

        # Concatenate all the attention heads
        # so that the last dimension summed up to model_size
        heads = tf.concat(heads, axis=2)
        heads = self.wo(heads)
        # heads has shape (batch, query_len, model_size)
        return heads


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        
        # One Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size, mask_zero=True)
        
        # num_layers Multi-Head Attention and Normalization layers
        self.attention = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        # num_layers FFN and Normalization layers
        self.dense_1 = [tf.keras.layers.Dense(model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

    def call(self, sequence):
        # sub_in = []
        # for i in range(sequence.shape[1]):
        #     # Compute the embedded vector
        #     # embed = self.embedding(tf.expand_dims(sequence[:, i], axis=1))
        #     embed = self.embedding(sequence[:, i, :])

        #     # Add positional encoding to the embedded vector
        #     sub_in.append(embed)

        # # Concatenate the result so that the shape is (batch_size, length, model_size)
        # sub_in = tf.concat(sub_in, axis=1)
        sub_in = sequence
        
        # We will have num_layers of (Attention + FFN)
        for i in range(self.num_layers):
            sub_out = self.attention[i](sub_in, sub_in)

            # Residual connection
            sub_out = sub_in + sub_out
            # Normalize the output
            sub_out = self.attention_norm[i](sub_out)

            # The FFN input is the output of the Multi-Head Attention
            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            # Add the residual connection
            ffn_out = ffn_in + ffn_out
            # Normalize the output
            ffn_out = self.ffn_norm[i](ffn_out)

            # Assign the FFN output to the next layer's Multi-Head Attention input
            sub_in = ffn_out
            
        # Return the result when done
        return ffn_out


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size, mask_zero=True)
        self.my_embedding = tf.keras.layers.Dense(model_size)
        self.attention_bot = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_bot_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_mid_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        
        self.dense_1 = [tf.keras.layers.Dense(model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, target_sequence, encoder_output):
        # EMBEDDING AND POSITIONAL EMBEDDING
        # embed_out = []
        # for i in range(target_sequence.shape[1]):
        #     # embed = self.embedding(tf.expand_dims(target_sequence[:, i], axis=1))
        #     embed = self.embedding(target_sequence[:, i, :])
        #     embed_out.append(embed)

        # embed_out = tf.concat(embed_out, axis=1)
        # bot_sub_in = embed_out

        # The target sequence is (bs, seqlength, 2), need it to have model_size as last.
        bot_sub_in = self.my_embedding(target_sequence)

        for i in range(self.num_layers):
            # BOTTOM MULTIHEAD SUB LAYER
            bot_sub_out = self.attention_bot[i](bot_sub_in, bot_sub_in)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)
            
            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out
            
            mid_sub_out = self.attention_mid[i](mid_sub_in, encoder_output)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out

        logits = self.dense(ffn_out)

        return logits


class Transformer(tf.keras.Model):
    def __init__(self, config_dict):
        super(Transformer, self).__init__()
        self.config_dict = config_dict
        self.feature_mask = tf.keras.layers.Masking(
            mask_value=0.,
            input_shape=(self.config_dict['video_pad_length'], self.config_dict['feature_dim']))
        self.target_mask = tf.keras.layers.Masking(
            mask_value=0.,
            input_shape=(self.config_dict['video_pad_length'], self.config_dict['nb_labels']))
        self.model_size = self.config_dict['model_size']
        self.nb_layers_enc = self.config_dict['nb_layers_enc']
        self.nb_layers_dec = self.config_dict['nb_layers_dec']
        self.nb_heads_enc = self.config_dict['nb_heads_enc']
        self.nb_heads_dec = self.config_dict['nb_heads_dec']
        self.model_size = self.config_dict['model_size']
        self.encoder = Encoder(vocab_size=self.config_dict['feature_dim'],
                               model_size=self.config_dict['model_size'],
                               num_layers=self.config_dict['nb_layers_enc'],
                               h=self.config_dict['nb_heads_enc'])
        self.decoder = Decoder(vocab_size=self.config_dict['nb_labels'],
                               model_size=self.config_dict['model_size'],
                               num_layers=self.config_dict['nb_layers_dec'],
                               h=self.config_dict['nb_heads_dec'])

    def call(self, input_features, target_sequence):
        input_features = self.feature_mask(input_features)
        target_sequence = self.target_mask(target_sequence)
        encoder_output = self.encoder(input_features)
        decoder_output = self.decoder(target_sequence, encoder_output)
        return decoder_output


def get_transformer_model(config_dict):
    input_features = tf.keras.layers.Input(shape=(config_dict['video_pad_length'], config_dict['feature_dim']))
    target_sequence = tf.keras.layers.Input(shape=(config_dict['video_pad_length'], config_dict['nb_labels']))

    transformer = Transformer(config_dict)

    decoder_output = transformer(input_features, target_sequence)

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


