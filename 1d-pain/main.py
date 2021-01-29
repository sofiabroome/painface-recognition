import tensorflow as tf
import wandb
import train

from config_dict import config_dict

def run():
    wandb.init(project="1d-pain")
    
    # model = get_dense_model(T=T)
    model = get_gru_model(T=T)
    model = get_identity_model(T=T)
    
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config_dict['lr'],
        decay_steps=40,
        decay_rate=0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=config_dict['lr'])
    
    train_dataset = construct_dataset(nb_pain=200, nb_nopain=200, T=T, base_level=base_level,
                          max_intensity_pain=max_intensity_pain, max_intensity_nopain=max_intensity_nopain,
                          max_length_pain=max_length_pain, max_length_nopain=max_length_nopain,
                          min_events_pain=min_events_pain,nb_events_pain=nb_events_pain,min_events_nopain=min_events_nopain,
                          nb_events_nopain=nb_events_nopain, batch_size=batch_size)
    
    val_dataset = construct_dataset(nb_pain=50, nb_nopain=50, T=T, base_level=base_level,
                          max_intensity_pain=max_intensity_pain, max_intensity_nopain=max_intensity_nopain,
                          max_length_pain=max_length_pain, max_length_nopain=max_length_nopain,
                          min_events_pain=min_events_pain, nb_events_pain=nb_events_pain,
                          min_events_nopain=min_events_nopain, nb_events_nopain=nb_events_nopain, batch_size=val_batch_size)
    
    train.train(train_dataset, val_dataset, model, config_dict)



if __name__ == '__main__':
    run()
