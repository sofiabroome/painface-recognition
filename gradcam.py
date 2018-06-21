from keras.layers import ConvLSTM2D, TimeDistributed, MaxPooling2D, Dense, Activation, Flatten, BatchNormalization, InputLayer
from keras.metrics import binary_accuracy as accuracy
from keras.objectives import binary_crossentropy
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.framework import ops
from keras.models import Model
import tensorflow as tf
import pandas as pd
import numpy as np
import tempfile
import keras

from image_processor import process_image
from helpers import find_between
import visualize_gradcam


# Replace vanila relu to guided relu to get guided backpropagation.
@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]),
                    tf.zeros(grad.get_shape()))

class CLSTMNetwork:
    def __init__(self, rgb, from_scratch, path=None):
        self.rgb = rgb
        self.path = path
        if from_scratch:
            self.build_from_scratch()
        else:
            self.m = self.build_from_saved_weights()

    def build_from_scratch(self):
        with tf.variable_scope('clstm1'):
            x = ConvLSTM2D(filters=6, kernel_size=(5,5), input_shape=(3,128,128,3),
                           batch_input_shape=(1,3,128,128,3), padding='same',
                           return_sequences=True)(self.rgb)
        x = TimeDistributed(MaxPooling2D())(x)
        x = BatchNormalization()(x)
        with tf.variable_scope('clstm2'):
            x = ConvLSTM2D(filters=6, kernel_size=(5,5), input_shape=(3,128,128,3),
                           padding='same',return_sequences=True)(x)
        x = TimeDistributed(MaxPooling2D())(x)
        x = BatchNormalization()(x)
        with tf.variable_scope('clstm3'):
            x = ConvLSTM2D(filters=6, kernel_size=(5,5), input_shape=(3,128,128,3),
                           padding='same',return_sequences=True)(x)
        x = TimeDistributed(MaxPooling2D())(x)
        x = BatchNormalization()(x)
        with tf.variable_scope('clstm4'):
            self.clstm4 = ConvLSTM2D(filters=6, kernel_size=(5,5), input_shape=(3,128,128,3),
                                     padding='same',return_sequences=True)(x)
        x = TimeDistributed(MaxPooling2D())(self.clstm4)
        x = BatchNormalization()(x)
        x = TimeDistributed(Flatten())(x)
        with tf.variable_scope('dense'):
            self.dense = Dense(2)(x)
        self.preds = Activation('sigmoid')(self.dense)

    def build_from_saved_weights(self):
        m = keras.models.load_model(self.path)
        self.clstm1 = m.layers[0](self.rgb)
        x = m.layers[1](self.clstm1)
        x = m.layers[2](x)
        self.clstm2 = m.layers[3](x)
        x = m.layers[4](self.clstm2)
        x = m.layers[5](x)
        self.clstm3 = m.layers[6](x)
        x = m.layers[7](self.clstm3)
        x = m.layers[8](x)
        self.clstm4 = m.layers[9](x)
        x = m.layers[10](self.clstm4)
        x = m.layers[11](x)
        x = m.layers[12](x)
        self.dense = m.layers[13](x)
        self.preds = m.layers[14](self.dense)
        return m
    
width = 128
height = 128
channels = 3

image_paths = ['data/jpg_128_128_2fps/horse_2/2_4/frame_000100.jpg', 
               'data/jpg_128_128_2fps/horse_2/2_4/frame_000101.jpg', 
               'data/jpg_128_128_2fps/horse_2/2_4/frame_000102.jpg', 
               'data/jpg_128_128_2fps/horse_2/2_4/frame_000103.jpg', 
               'data/jpg_128_128_2fps/horse_2/2_4/frame_000104.jpg', 
               'data/jpg_128_128_2fps/horse_2/2_4/frame_000105.jpg', 
               'data/jpg_128_128_2fps/horse_2/2_4/frame_000106.jpg', 
               'data/jpg_128_128_2fps/horse_2/2_4/frame_000107.jpg', 
               'data/jpg_128_128_2fps/horse_2/2_4/frame_000108.jpg', 
               'data/jpg_128_128_2fps/horse_2/2_4/frame_000109.jpg']

# image_paths = ['data/jpg_128_128_2fps/horse_2/2_1a/frame_000500.jpg', 
#                'data/jpg_128_128_2fps/horse_2/2_1a/frame_000501.jpg', 
#                'data/jpg_128_128_2fps/horse_2/2_1a/frame_000502.jpg', 
#                'data/jpg_128_128_2fps/horse_2/2_1a/frame_000503.jpg', 
#                'data/jpg_128_128_2fps/horse_2/2_1a/frame_000504.jpg', 
#                'data/jpg_128_128_2fps/horse_2/2_1a/frame_000505.jpg', 
#                'data/jpg_128_128_2fps/horse_2/2_1a/frame_000506.jpg', 
#                'data/jpg_128_128_2fps/horse_2/2_1a/frame_000507.jpg', 
#                'data/jpg_128_128_2fps/horse_2/2_1a/frame_000508.jpg', 
#                'data/jpg_128_128_2fps/horse_2/2_1a/frame_000509.jpg']

nb_ims = len(image_paths)
horse_id = find_between(image_paths[0], 'fps/', '/')
video_id = find_between(image_paths[0], horse_id + '/', '/frame')
metadata = pd.read_csv('videos_overview_missingremoved.csv', sep=';')
pain = metadata[metadata['Video_id']==video_id]['Pain'].values[0]

# best_model_path = 'models/BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_6_CONVfilters_5_test5d.h5'
best_model_path = 'models/BEST_MODEL_convolutional_LSTM_adadelta_LSTMunits_32_CONVfilters_None_jpg128_2fps_val4_t1_seq10ss10_4hl_32ubs16_no_aug_june.h5'

if pain:
    label_onehot = np.array([1 if i == 1 else 0 for i in range(2)])
else:
    label_onehot = np.array([1 if i == 0 else 1 for i in range(2)])

label_onehot = label_onehot.reshape(1,1,-1)
batch_label = np.asarray(nb_ims*[label_onehot]).reshape(1, nb_ims, -1)

imgs = []
for img_path in image_paths:
    img = process_image(img_path, (width, height, channels))
    img = img.reshape((1, 1, width, height, channels))
    imgs.append(img)

batch_img = np.concatenate(imgs, axis=1)
batch_size = nb_ims

# DEFINE GRAPH

from keras import backend as K

images = tf.placeholder(tf.float32, [1, batch_size, width, height, channels])
labels = tf.placeholder(tf.float32, [1, batch_size, 2])

clstm_model = CLSTMNetwork(images, from_scratch=0, path=best_model_path)

sess = K.get_session()  # Grab the Keras session where the weights are initialized.

cost = (-1) * tf.reduce_sum(tf.multiply(labels, tf.log(clstm_model.preds)), axis=1)

y_c = tf.reduce_sum(tf.multiply(clstm_model.dense, labels), axis=1)

target_conv_layer = clstm_model.clstm3 # Choose which CLSTM-layer to study

target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]

gb_grad = tf.gradients(cost, images)[0]  # Guided backpropagation back to input layer


with sess.as_default():
    prob = sess.run(clstm_model.preds,
                    feed_dict={images: batch_img,
                    K.learning_phase(): 0})

    print(prob)

    gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = \
        sess.run([gb_grad, target_conv_layer, target_conv_layer_grad],
                  feed_dict={images: batch_img,
                  labels: batch_label,
                  K.learning_phase(): 0})

    for i in range(batch_size):
        print(prob[0,i,:])
        visualize_gradcam.visualize(batch_img[:,i,:,:],
                                    target_conv_layer_value[:,i,:,:],
                                    target_conv_layer_grad_value[:,i,:,:],
                                    gb_grad_value[:,i,:,:], number=i)

