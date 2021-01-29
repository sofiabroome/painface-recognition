import sys
sys.path.append('..')

import tensorflow as tf


def get_sparse_pain_loss(y_batch, preds_batch):
    batch_size = y_batch.shape[0]  # last batch may be smaller

    def get_mil_loss(kmax_scores):
        kmax_distribution = tf.keras.layers.Activation('softmax')(kmax_scores)
        mils = 0
        for sample_index in range(batch_size):
            label_index = tf.argmax(y_batch[sample_index, :])  # Take first (video-level label)
            mil = tf.math.log(kmax_distribution[sample_index, label_index])
            mils += mil
        return mils

    kmax_scores = get_k_max_scores_per_class(y_batch, preds_batch, batch_size)
    mil = get_mil_loss(kmax_scores)
    batch_indicator_nopain = tf.cast(y_batch[:, 0], dtype=tf.float32)
    batch_indicator_pain = tf.cast(y_batch[:, 1], dtype=tf.float32)
    if config_dict['tv_weight_nopain'] == 0:
        tv_nopain = 0
    else:
        tv_nopain = config_dict['tv_weight_nopain'] * tf.reduce_sum(
            batch_indicator_nopain * batch_calc_TV_norm(preds_batch[:, 0],
                                                        y_batch))
    if config_dict['tv_weight_pain'] == 0:
        tv_pain = 0
    else:
        tv_pain = config_dict['tv_weight_pain'] * tf.reduce_sum(
            batch_indicator_pain * batch_calc_TV_norm(preds_batch[:, 1],
                                                  y_batch))
    total_loss = tv_nopain + tv_pain - mil
                                                                                                                                                        
    return total_loss, tv_pain, tv_nopain, mil

def get_k_max_scores_per_class(y_batch, preds_batch, batch_size):                                                           
    kmax_scores = []
    for sample_index in range(batch_size):
        sample_class_kmax_scores = []
        seq_length = 144
        k = tf.cast(tf.math.ceil(config_dict['k_mil_fraction'] * tf.cast(seq_length, dtype=tf.float32)), dtype=tf.int32)
        # print('\n', k, seq_length)
        for class_index in range(config_dict['nb_labels']):
            preds_nopad = preds_batch[sample_index, :, class_index]
            k_preds, indices = tf.math.top_k(preds_nopad, k)
            sample_class_kmax_score = tf.cast(1/k, dtype=tf.float32) * tf.reduce_sum(k_preds)
            sample_class_kmax_scores.append(sample_class_kmax_score)
        kmax_scores.append(sample_class_kmax_scores)
    kmax_scores = tf.convert_to_tensor(kmax_scores)
    return kmax_scores


def batch_calc_TV_norm(batch_vectors, y_batch, p=3, q=3):
    """"
    Calculates the Total Variational Norm by summing the differences of the values
    in between the different positions in the mask.
    p=3 and q=3 are defaults from the paper.
    """
    val = tf.cast(0, dtype=tf.float32)
    batch_size = batch_vectors.shape[0]
    batch_length = batch_vectors.shape[1]
    vals = tf.TensorArray(tf.float32, size=batch_size)
    for vector_index in range(batch_size):
        vector = batch_vectors[vector_index]
        for u in range(1, batch_length - 1):
            val += tf.abs(vector[u - 1] - vector[u]) ** p
            val += tf.abs(vector[u + 1] - vector[u]) ** p
        val = val ** (1 / p)
        val = val ** q
        vals = vals.write(vector_index, val)
    return vals.stack()

def evaluate_sparse_pain(y_batch, preds_batch):
    batch_size = y_batch.shape[0]
    kmax_scores = get_k_max_scores_per_class(y_batch, preds_batch, batch_size)
    batch_class_distribution = tf.keras.layers.Activation('softmax')(kmax_scores)
    return batch_class_distribution

def mask_out_padding_predictions(preds_batch, length, batch_size, pad_length, one):                                                                                                                                                      
    zeros = tf.zeros_like(preds_batch)

    mask_tensor = tf.TensorArray(tf.float32, size=batch_size)

    if one is None:  # Cannot recreate tf.Variables in each step.
        one = tf.Variable(tf.ones([pad_length, 1]))

    for u in range(batch_size):
        one.assign(tf.ones([pad_length, 1]))
        indices = tf.where([(i >= length[u]) for i in range(pad_length)])
        # Put zeros in the mask where the sum of the y-labels is 0
        hej = tf.gather(zeros[0,:,0], indices)
        mask = tf.compat.v1.scatter_nd_update(one, indices, hej)
        # Need mask for both classes: (pad_length x nb_labels)
        masks = tf.stack([mask, mask], axis=1)
        mask_tensor = mask_tensor.write(u, masks)
    mask_tensor = mask_tensor.stack()
    mask_tensor = tf.reshape(mask_tensor, preds_batch.shape)
    preds_batch = tf.keras.layers.multiply([preds_batch, mask_tensor])
    return preds_batch

def train(train_dataset, val_dataset, model, config_dict):
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()
    one = None

    @tf.function
    def train_step(x, y, length, one):
        with tf.GradientTape() as tape:
            preds_seq = model(x)
            # print(y.shape, preds_seq.shape)
            preds_seq = mask_out_padding_predictions(preds_seq, length, batch_size, T, one)
            
            sparse_loss, tv_p, tv_np, mil = get_sparse_pain_loss(y, preds_seq)
            preds_mil = evaluate_sparse_pain(y, preds_seq)
            loss = sparse_loss
            # loss = loss_fn(y, preds_seq)
        
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y, preds_mil)
        # train_acc_metric.update_state(y, preds_seq)
        return loss
        
    @tf.function
    def val_step(x, y, length, one):
        preds_seq = model(x)
        preds_seq = mask_out_padding_predictions(preds_seq, length, val_batch_size, T, one)
        sparse_loss, tv_p, tv_np, mil = get_sparse_pain_loss(y, preds_seq)
        preds_mil = evaluate_sparse_pain(y, preds_seq)
        loss = sparse_loss
        val_acc_metric.update_state(y, preds_mil)
    #     loss = loss_fn(y, preds_seq)
    #     val_acc_metric.update_state(y, preds_seq)
        return loss

    for epoch in range(config_dict['epochs']):
        for x, y, length in train_dataset:
            train_loss = train_step(x, y, length, one)
            wandb.log({'train_loss': train_loss})
        train_acc = train_acc_metric.result()
        wandb.log({'train_acc': train_acc})
        
            
        for x, y, length in val_dataset:
            val_loss = val_step(x, y, length, one)
            wandb.log({'val_loss': val_loss})
        val_acc = val_acc_metric.result()
        wandb.log({'val_acc': val_acc})
        if epoch % 20 == 0:
            print('Training acc over epoch %d: %.4f' % (epoch, float(train_acc),))
            print("Validation acc: %.4f" % (float(val_acc),))
