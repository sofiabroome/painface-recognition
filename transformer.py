import tensorflow as tf

# Encoder, decoder and mh-attention classes from
# https://trungtran.io/2019/04/29/create-the-transformer-with-tensorflow-2-0/


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.key_size = model_size // h
        self.h = h
        self.wq = tf.keras.layers.Dense(model_size) #[tf.keras.layers.Dense(key_size) for _ in range(h)]
        self.wk = tf.keras.layers.Dense(model_size) #[tf.keras.layers.Dense(key_size) for _ in range(h)]
        self.wv = tf.keras.layers.Dense(model_size) #[tf.keras.layers.Dense(value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size)
        # self.query_size = model_size // h
        # self.key_size = model_size // h
        # self.value_size = model_size // h
        # self.h = h
        # self.wq = [tf.keras.layers.Dense(self.query_size) for _ in range(h)]
        # self.wk = [tf.keras.layers.Dense(self.key_size) for _ in range(h)]
        # self.wv = [tf.keras.layers.Dense(self.value_size) for _ in range(h)]
        # self.wo = tf.keras.layers.Dense(model_size)

    def call(self, decoder_output, encoder_output):
        # query has shape (batch, query_len, model_size)
        # value has shape (batch, value_len, model_size)

        query = self.wq(decoder_output)
        key = self.wk(encoder_output)
        value = self.wv(encoder_output)
        
        # Split for multihead attention

        query_seq_length = query.shape[1]
        key_seq_length = key.shape[1]
        value_seq_length = value.shape[1]

        query = tf.reshape(query, [-1, query_seq_length, self.h, self.key_size])
        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.reshape(key, [-1, key_seq_length, self.h, self.key_size])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.reshape(value, [-1, value_seq_length, self.h, self.key_size])
        value = tf.transpose(value, [0, 2, 1, 3])
        
        score = tf.matmul(query, key, transpose_b=True)
        score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, dtype=tf.float32))

        # if mask is not None:
        #     score *= mask
        #     score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)

        alignment = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(alignment, value)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [-1, query_seq_length, self.key_size * self.h])
 
        heads = self.wo(context)

        # # heads = []
        # # for i in range(self.h):
        # #     score = tf.matmul(self.wq[i](query), self.wk[i](value), transpose_b=True)

        # #     # Here we scale the score as described in the paper
        # #     score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
        # #     # score has shape (batch, query_len, value_len)

        # #     alignment = tf.nn.softmax(score, axis=2)
        # #     # alignment has shape (batch, query_len, value_len)

        # #     head = tf.matmul(alignment, self.wv[i](value))
        # #     # head has shape (batch, decoder_len, value_size)
        # #     heads.append(head)

        # # # Concatenate all the attention heads
        # # # so that the last dimension summed up to model_size
        # # heads = tf.concat(heads, axis=2)
        # # heads = self.wo(heads)

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
        self.my_embedding = tf.keras.layers.Dense(model_size)
        
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
        sub_in = self.my_embedding(sequence)
        # sub_in = sequence
        
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
        self.feature_mask = tf.keras.layers.Masking(
            mask_value=0.,
            input_shape=(config_dict['video_pad_length'], config_dict['feature_dim']))
        self.target_mask = tf.keras.layers.Masking(
            mask_value=0.,
            input_shape=(config_dict['video_pad_length'], config_dict['nb_labels']))
        self.model_size = config_dict['model_size']
        self.nb_layers_enc = config_dict['nb_layers_enc']
        self.nb_layers_dec = config_dict['nb_layers_dec']
        self.nb_heads_enc = config_dict['nb_heads_enc']
        self.nb_heads_dec = config_dict['nb_heads_dec']
        self.model_size = config_dict['model_size']
        self.encoder = Encoder(vocab_size=config_dict['feature_dim'],
                               model_size=config_dict['model_size'],
                               num_layers=config_dict['nb_layers_enc'],
                               h=config_dict['nb_heads_enc'])
        self.decoder = Decoder(vocab_size=config_dict['nb_labels'],
                               model_size=config_dict['model_size'],
                               num_layers=config_dict['nb_layers_dec'],
                               h=config_dict['nb_heads_dec'])

    def call(self, input_features, target_sequence):
        input_features = self.feature_mask(input_features)
        target_sequence = self.target_mask(target_sequence)
        encoder_output = self.encoder(input_features)
        decoder_output = self.decoder(target_sequence, encoder_output)
        return decoder_output

