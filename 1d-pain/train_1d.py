import sys
sys.path.append('..')

import tensorflow as tf
import wandb
import test_and_eval
import train


def train_1d(train_dataset, val_dataset, model, optimizer, config_dict):
    train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    val_acc_metric = tf.keras.metrics.BinaryAccuracy()

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[config_dict['batch_size'], config_dict['T']], dtype=tf.float32),
        tf.TensorSpec(shape=[config_dict['batch_size'], 2], dtype=tf.int32),
        tf.TensorSpec(shape=[config_dict['batch_size'],], dtype=tf.int32)))
    def train_step(x, y, length):
        with tf.GradientTape() as tape:
            preds_seq = model(x)
            preds_seq = train.mask_out_padding_predictions(
                preds_seq, length, config_dict['batch_size'], config_dict['T'])
            
            sparse_loss, tv_p, tv_np, mil = train.get_sparse_pain_loss(
                y, preds_seq, length, config_dict)
            preds_mil = test_and_eval.evaluate_sparse_pain(
                preds_seq, length, config_dict)
            loss = sparse_loss
            # loss = loss_fn(y, preds_seq)
        
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y, preds_mil)
        # train_acc_metric.update_state(y, preds_seq)
        return loss
        
    @tf.function(input_signature=(
        tf.TensorSpec(shape=[config_dict['val_batch_size'], config_dict['T']], dtype=tf.float32),
        tf.TensorSpec(shape=[config_dict['val_batch_size'], 2], dtype=tf.int32),
        tf.TensorSpec(shape=[config_dict['val_batch_size'],], dtype=tf.int32)))
    def val_step(x, y, length):
        preds_seq = model(x)
        preds_seq = train.mask_out_padding_predictions(
            preds_seq, length, config_dict['val_batch_size'], config_dict['T'])
        sparse_loss, tv_p, tv_np, mil = train.get_sparse_pain_loss(
            y, preds_seq, length, config_dict)
        preds_mil = test_and_eval.evaluate_sparse_pain(preds_seq, length, config_dict)
        loss = sparse_loss
        val_acc_metric.update_state(y, preds_mil)
    #     loss = loss_fn(y, preds_seq)
    #     val_acc_metric.update_state(y, preds_seq)
        return loss

    for epoch in range(config_dict['epochs']):
        wandb.log({'epoch': epoch})
        for x, y, length in train_dataset:
            train_loss = train_step(x, y, length)
            wandb.log({'train_loss': train_loss})
        train_acc = train_acc_metric.result()
        wandb.log({'train_acc': train_acc})
        
        for x, y, length in val_dataset:
            val_loss = val_step(x, y, length)
            wandb.log({'val_loss': val_loss})
        val_acc = val_acc_metric.result()
        wandb.log({'val_acc': val_acc})
        if epoch % 20 == 0:
            print('Training acc over epoch %d: %.4f' % (epoch, float(train_acc),))
            print("Validation acc: %.4f" % (float(val_acc),))

