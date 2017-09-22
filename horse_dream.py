import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
import numpy as np
import keras

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def get_noise(batch_size, seq_length, width, height, channels):
    # noise = np.random.normal(0, 1, (batch_size, width, height))
    noise = np.random.random((batch_size, seq_length, height, width, channels))*1
    return noise

if __name__ == '__main__':

    with tf.Session() as sess:

        K.set_learning_phase(1)

        model_name = '2tream_5d'
        model_path = 'models/BEST_MODEL_2stream_5d_adam_LSTMunits_64_CONVfilters_16_val4_02finaldropout_seq60_t3_1conv4lstmlayers.h5'
        model = None
        if model_name == "bidir":
            model = model
        else:
            model = keras.models.load_model(model_path)

        input = model.input

        # Maximizing class
        output_index = 1
        # layer_dict = dict([(layer.name, layer) for layer in model.layers])
        # For testing filters
        layer_dict = dict([(layer.name, layer) for layer in model.layers if not 'lstm' in layer.name])

        loss = K.mean(model.output[:, :, output_index])

        batch_size = 1
        seq_length = 1
        width = 320
        height = 180
        channels = 3

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input[0], input[1]], [loss, grads])
        num_iterations = 500
        step = 0.1
        input_data = get_noise(batch_size, seq_length, width, height, channels)

        f, axes = plt.subplots(nrows=2, ncols=1)

        rgb_reshaped = np.reshape(input_data, (width, height, channels))

        axes[0].imshow(rgb_reshaped)
        loss_value = 0
        while loss_value < 0.998:
            loss_value, grads_value = iterate([input_data, input_data])
            print(loss_value)
            # print(grads_value)
            input_data += grads_value * step
        import ipdb; ipdb.set_trace()
        input_data = np.reshape(input_data[0], (width, height, channels))

        axes[1].imshow(input_data)
        plt.show()
