import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np
import arg_parser
import helpers
import models
import train_1d
import wandb
import data


def run():
    if config_dict['model_name'] == 'transformer':
        model = models.get_transformer_model(config_dict)
    if config_dict['model_name'] == 'gru':
        model = models.get_gru_model(config_dict)
    if config_dict['model_name'] == 'gru_attention':
        model = models.get_gru_attention_model(config_dict)
    if config_dict['model_name'] == 'dense':
        model = models.get_dense_model(config_dict)
    if config_dict['model_name'] == 'id':
        model = models.get_identity_model(config_dict)

    # loss_fn = tf.keras.losses.BinaryCrossentropy()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config_dict['lr'],
        decay_steps=40,
        decay_rate=0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=config_dict['lr'])

    rng = np.random.default_rng(seed=3) 

    train_dataset = data.construct_dataset(config_dict['nb_pain_train'], config_dict['nb_nopain_train'],
                                           config_dict['batch_size'], config_dict, rng)
    
    val_dataset = data.construct_dataset(config_dict['nb_pain_val'], config_dict['nb_nopain_val'],
                                         config_dict['val_batch_size'], config_dict, rng)
    
    train_1d.train_1d(train_dataset, val_dataset, model, optimizer, config_dict)


if __name__ == '__main__':
    arg_parser = arg_parser.ArgParser(len(sys.argv))
    args = arg_parser.parse()

    config_dict_module = helpers.load_module(args.config_file)
    config_dict = config_dict_module.config_dict
    config_dict['nb_pain_train'] = args.nb_pain_train
    config_dict['nb_nopain_train'] = args.nb_nopain_train
    config_dict['nb_pain_val'] = args.nb_pain_val
    config_dict['nb_nopain_val'] = args.nb_nopain_val
    config_dict['job_identifier'] = args.job_identifier

    if args.test_run == 1:
        config_dict['epochs'] = 1

    print(config_dict)
    print('Job identifier: ', args.job_identifier) 
    wandb.init(project='1d-pain', config=config_dict)
    # wandb.save('train_1d.py')
    # wandb.save('../train.py')

    run()
