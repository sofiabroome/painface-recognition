# coding: utf-8

import sys
sys.path.append('../')

from data_handler import get_video_id_stem_from_path
from data_handler import get_video_id_from_path
from data_handler import get_video_id_from_frame_path
import visualization
import visualization.visualize_gradcam
import visualization.gradcam as gradcam
from helpers import process_image
from keras.utils import np_utils
from keras.layers import Activation
import skvideo.io
import pandas as pd
import data_handler
import numpy as np
import keras
import cv2
import os


class ArgsProxy:
    def __init__(self, data_path, of_path, input_height, input_width,
                 seq_length, seq_stride, batch_size, nb_labels,
                 aug_flip, aug_crop, aug_light):
        self.data_path = data_path
        self.of_path = of_path
        self.input_height = input_height
        self.input_width = input_width
        self.seq_length = seq_length
        self.seq_stride = seq_stride
        self.batch_size = batch_size
        self.nb_labels = nb_labels
        self.aug_flip = aug_flip
        self.aug_crop = aug_crop
        self.aug_light = aug_light


def read_or_create_subject_dfs(dh, args, subject_ids):
    """
    Read or create the per-subject dataframes listing
    all the frame paths and corresponding labels and metadata.
    :param dh: DataHandler
    :return: [pd.Dataframe]
    """
    subject_dfs = []
    for subject_id in subject_ids:
        print(args.data_path)
        subject_csv_path = args.data_path + subject_id + '.csv'
        if os.path.isfile(subject_csv_path):
            sdf = pd.read_csv(subject_csv_path)
        else:
            print('Making a DataFrame for subject id: ', subject_id)
            sdf = dh.subject_to_df(subject_id)
            sdf.to_csv(path_or_buf=subject_csv_path)
        subject_dfs.append(sdf)
    return subject_dfs

def read_or_create_subject_rgb_and_OF_dfs(dh,
                                          args,
                                          subject_ids,
                                          subject_dfs):
    """
    Read or create the per-subject optical flow files listing
    all the frame paths and labels.
    :param dh: DataHandler object
    :param subject_dfs: [pd.DataFrame]
    :return: [pd.DataFrame]
    """
    subject_rgb_OF_dfs = []
    for ind, subject_id in enumerate(subject_ids):
        subject_of_csv_path = dh.of_path + str(subject_id) + '.csv'
        if os.path.isfile(subject_of_csv_path):
            sdf = pd.read_csv(subject_of_csv_path)
        else:
            print('Making a DataFrame for subject id: ', subject_id)
            sdf = dh.save_OF_paths_to_df(subject_id,
                                         subject_dfs[ind])
            sdf.to_csv(path_or_buf=subject_of_csv_path)
        subject_rgb_OF_dfs.append(sdf)
    return subject_rgb_OF_dfs


def get_sequence(args, subject_dfs, subject=-1, video=None, start_index=None):
    """
    :param subject: int [0,5]
    "param video: str '5_5b'"
    """
    if subject==-1 and video:
        print('Need to provide both subject and video ID to get sequence function.')
    if subject == -1:
        random_subject = np.random.randint(0,6)
        df = subject_dfs[random_subject]
    else:
        print('Chose subject ', subject)
        df = subject_dfs[subject]
    if video:
        df = df[df['Video_ID'] == video]
    if start_index is None:
        random_start_index = np.random.randint(0, len(df))
        start_index = random_start_index
    print('Start index in subject dataframe: ', start_index)
    end_index = start_index + args.seq_length
    
    sequence_df = df.iloc[start_index:end_index]
    sequence_df.reset_index(drop=True, inplace=True)
    
    vid_id_first = get_video_id_from_frame_path(sequence_df.loc[0]['Path'])
    vid_id_last = get_video_id_from_frame_path(sequence_df.loc[args.seq_length-1]['Path'])

    assert(vid_id_first == vid_id_last)
    
    return sequence_df

def make_video_from_frames(frames, path):
    """
    :param frames: [np.array]
    :param path: str, f ex 'output.avi'
    """
    height, width, channels = frames[0,0].shape
    seq_length = frames.shape[1]

    # video = cv2.VideoWriter(path, -1, 1, (width, height))
    # for i in range(seq_length):
    #     video.write(frames[0][i])
    # 
    # cv2.destroyAllWindows()
    # video.release()

    frames = frames.astype(np.uint8)
    inputdict = {'-r': '2'}
    outputdict = {'-r': '2'}
    writer = skvideo.io.FFmpegWriter(path, inputdict=inputdict, outputdict=outputdict)
    print(frames.shape)
    for i in range(seq_length):
            writer.writeFrame(frames[0][i])
    writer.close()


def data_for_one_random_sequence_two_stream(args, subject_dfs, computer, subject=None, start_index=None):
    sequence_df = get_sequence(args, subject_dfs, subject=subject, start_index=start_index)
    print(sequence_df[['Path', 'Pain']])

    image_paths = sequence_df['Path'].values
    of_paths = sequence_df['OF_Path'].values
    y = sequence_df['Pain'].values
    
    label_onehot = np_utils.to_categorical(y, num_classes=args.nb_labels)
    batch_label = label_onehot.reshape(args.batch_size, args.seq_length, -1)

    batch_img = np.concatenate(read_images_and_return_list(args, image_paths, computer), axis=1)
    batch_flow = np.concatenate(read_images_and_return_list(args, of_paths, computer), axis=1)
    return batch_img, batch_flow, batch_label


def data_for_one_random_sequence_4D(args, subject_dfs, computer, subject=None, start_index=None):
    sequence_df = get_sequence(args, subject_dfs, subject=subject, start_index=start_index)
    print(sequence_df[['Path', 'Pain']])

    image_paths = sequence_df['Path'].values
    y = sequence_df['Pain'].values
    
    label_onehot = np_utils.to_categorical(y, num_classes=args.nb_labels)
    batch_label = label_onehot.reshape(args.batch_size, -1)

    batch_img = np.concatenate(read_images_and_return_list(args, image_paths, computer), axis=1)
    batch_img = np.reshape(batch_img, (args.batch_size, args.input_width, args.input_height, 3))
    return batch_img, batch_label

def data_for_one_random_sequence_5D(args, subject_dfs, computer, subject=None, start_index=None):
    sequence_df = get_sequence(args, subject_dfs, subject=subject, start_index=start_index)
    print(sequence_df[['Path', 'Pain']])
    image_paths = sequence_df['Path'].values
    y = sequence_df['Pain'].values
    
    label_onehot = np_utils.to_categorical(y, num_classes=args.nb_labels)
    batch_label = label_onehot.reshape(args.batch_size, args.seq_length, -1)

    batch_img = np.concatenate(read_images_and_return_list(args, image_paths, computer), axis=1)
    batch_img = np.reshape(batch_img, (args.batch_size, args.seq_length, args.input_width, args.input_height, 3))
    return batch_img, batch_label

def read_images_and_return_list(args, paths, computer='hg'):
    list_to_return = []
    for p in paths:
        if computer == 'hg':
            p = '/home/sofia/Documents/painface-recognition/' + p
        elif computer == 'local':
            p = '/Users/sbroome/Documents/EquineML/painface-recognition/' + p
        else:
            p = '/home/sbroome/dev/painface-recognition/' + p
        img = process_image(p, (args.input_width, args.input_height, 3))
        img = img.reshape((1,1,args.input_width, args.input_height, 3))
        list_to_return.append(img)
    return list_to_return


class RodriguezNetwork:
    def __init__(self, rgb, path=None):
        self.rgb = rgb
        self.path = path
        self.m = self.build_from_saved_weights()

    def build_from_saved_weights(self):
        m = keras.models.load_model(self.path)
        x = m.layers[0](self.rgb)
        self.timedist_vgg = m.layers[1](self.rgb)
        x = m.layers[2](self.timedist_vgg)
        self.dense_1 = m.layers[3](x)
        self.lstm = m.layers[4](self.dense_1)
        self.preds = m.layers[5](self.lstm)
        return m


class InceptionNetwork:
    def __init__(self, rgb, from_scratch, path=None):
        self.rgb = rgb
        self.path = path
        self.m = self.build_from_saved_weights()

    def build_from_saved_weights(self):
        print('Loading model...')
        m = keras.models.load_model(self.path)

        print('Finished loading model. Building layers...')
        # RGB-stream
        x = m.layers[0](self.rgb)
        for i in range(1,17):
            print(i)
            print(m.layers[i])
            x = m.layers[i](x)
        mp = m.layers[17](x)

        c = m.layers[18](mp)
        bn = m.layers[19](c)
        a = m.layers[20](bn)  # output 64

        import pdb; pdb.set_trace()
        print(m.layers[21])
        b55 = m.layers[21](mp)  # output 48
        print(m.layers[22])
        o96 = m.layers[22](a) # output 96
        o48 = m.layers[23](b55)  # output 48

        o96 = m.layers[24](o96)
        b55 = m.layers[25](o48)
        b55 = m.layers[26](o96)

        b33dbl = m.layers[27](mp)
        b33dbl = m.layers[28](b33dbl)
        b33dbl = m.layers[29](b33dbl)

        b33dbl = m.layers[30](b33dbl)
        b33dbl = m.layers[31](b33dbl)
        b33dbl = m.layers[32](b33dbl)

        b33dbl = m.layers[33](b33dbl)
        b33dbl = m.layers[34](b33dbl)
        b33dbl = m.layers[35](b33dbl)

        branch_pool = m.layers[36](mp)
        bp = m.layers[37](branch_pool)
        bp = m.layers[38](bp)
        bp = m.layers[39](bp)

        for i in range(20,299):
            print(i)
            print(m.layers[i])
            x = m.layers[i](x)
        import pdb; pdb.set_trace()
        self.lastconv = m.layers[299](x)
        x = m.layers[300](self.lastconv)
        for i in range(301,313):
            x = m.layers[i](x)
        
        self.dense = m.layers[313](x)
        self.preds = Activation('sigmoid')(self.dense)
        return m

class TwoStreamCLSTMNetwork:
    def __init__(self, rgb, optical_flow, from_scratch, path=None):
        self.rgb = rgb
        self.optical_flow = optical_flow
        self.path = path
        self.m = self.build_from_saved_weights()

    def build_from_saved_weights(self):
        m = keras.models.load_model(self.path)
        
        # RGB-stream
        x = m.layers[0](self.rgb)
        self.clstm1_rgb = m.layers[2].layers[0](x)
        x = m.layers[2].layers[1](self.clstm1_rgb)
        x = m.layers[2].layers[2](x)
        x = m.layers[2].layers[3](x)
        x = m.layers[2].layers[4](x)
        x = m.layers[2].layers[5](x)
        x = m.layers[2].layers[6](x)
        x = m.layers[2].layers[7](x)
        x = m.layers[2].layers[8](x)
        self.clstm4_rgb = m.layers[2].layers[9](x)
        x = m.layers[2].layers[10](self.clstm4_rgb)
        x = m.layers[2].layers[11](x)
        self.rgb_stream = m.layers[2].layers[12](x)
        
        # FLOW-stream
        y = m.layers[1](self.optical_flow)
        y = m.layers[3].layers[0](y)
        y = m.layers[3].layers[1](y)
        y = m.layers[3].layers[2](y)
        y = m.layers[3].layers[3](y)
        y = m.layers[3].layers[4](y)
        y = m.layers[3].layers[5](y)
        y = m.layers[3].layers[6](y)
        y = m.layers[3].layers[7](y)
        y = m.layers[3].layers[8](y)
        self.clstm4_of = m.layers[3].layers[9](y)
        y = m.layers[3].layers[10](self.clstm4_of)
        y = m.layers[3].layers[11](y)
        self.of_stream = m.layers[3].layers[12](y)
        
        self.merge = m.layers[4]([self.rgb_stream, self.of_stream])
        x = m.layers[5](self.merge)
        self.dense = m.layers[6](x)
        self.preds = Activation('sigmoid')(self.dense)
        return m


def create_graph_for_clstm(batch_size, args, channels, best_model_path, two_stream=True):    
    from keras import backend as K
    import tensorflow as tf
        
    images = tf.placeholder(tf.float32, [batch_size, args.seq_length, args.input_width, args.input_height, channels], name='images')
    flows = tf.placeholder(tf.float32, [batch_size, args.seq_length, args.input_width, args.input_height, channels], name='flows')
    labels = tf.placeholder(tf.float32, [batch_size, args.seq_length, 2], name='labels')
    
    if two_stream:
        clstm_model = TwoStreamCLSTMNetwork(images, flows, from_scratch=0, path=best_model_path)
    else:
        clstm_model = CLSTMNetwork(images, from_scratch=0, path=best_model_path)
        
    sess = K.get_session()  # Grab the Keras session where the weights are initialized.
    
    cost = (-1) * tf.reduce_sum(tf.multiply(labels, tf.log(clstm_model.preds)), axis=1)
        
    y_c = tf.reduce_sum(tf.multiply(clstm_model.dense, labels), axis=1)
        
    if two_stream:
        # target_conv_layer = clstm_model.clstm1_rgb # Choose which CLSTM-layer to study
        target_conv_layer = clstm_model.merge # Choose which CLSTM-layer to study
    else:
        target_conv_layer = clstm_model.clstm4 # Choose which CLSTM-layer to study
        
    target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]
        
    gb_grad = tf.gradients(cost, [images, flows])[0]  # Guided backpropagation back to input layer
    return sess, clstm_model


def plot_sequences(rgb, X_seq_list, flipped, cropped, shaded,
                      seq_index, batch_index, window_index):
    rows = 4
    cols = 10
    f, axarr = plt.subplots(rows, cols, figsize=(20,10))
    for i in range(0, rows):
        for j in range(0, cols):
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
            # axarr[i, j].set_aspect('equal')
            if i == 0:
                im = X_seq_list[j]
                im /= 255
                axarr[i, j].imshow(im)
            elif i == 1:
                im = flipped[j]
                im /= 255
                axarr[i, j].imshow(im)
            elif i == 2:
                im = cropped[j]
                im /= 255
                axarr[i, j].imshow(im)
            else:
                im = shaded[j]
                im /= 255
                axarr[i, j].imshow(im)
    plt.tick_params(axis='both', which='both', bottom='off', left='off')
    f.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.axis('off')
    # plt.tight_layout()
    plt.savefig('seq_{}_batch_{}_wi_{}_part_{}_rgb_{}.png'.format(seq_index,
                                                           batch_index,
                                                           window_index,
                                                           partition,
                                                           rgb))
    plt.close()

def visualize_overlays_4D(images, conv_outputs, conv_grads, flows=None):

    from skimage.transform import resize
    from matplotlib import pyplot as plt

    if flows is not None:
        nb_rows = 3
        fig_width = 23
        fig_height = 7
    else:
        nb_rows = 2
        fig_width = 22
        fig_height = 4.5

    image = images[0,:,:,:]
    if flows is not None:
        flow = flows[0,:,:,:]
        flow = flow.astype(float)
        flow -= np.min(flow)
        flow /= flow.max()

    output = conv_outputs[0,:,:,:]           # [7,7,512]
    grads_val = conv_grads[0,:,:,:]          # [7,7,512]
    import pdb; pdb.set_trace()
    # print("grads_val shape:", grads_val.shape)
    weights = np.mean(grads_val, axis = (0,1)) # alpha_k, [512]
    cam = np.zeros(output.shape[0 : 2], dtype = np.float32) # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) # scale 0 to 1.0
    cam = resize(cam, (320,180), preserve_range=True)

    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()

    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    # if im == 0:
    #     fig = plt.figure(figsize=(fig_width, fig_height))    
    # ax = fig.add_subplot(nb_rows,10,im+1)
    # ax.set_xticks([])
    # ax.set_yticks([])
    imgplot = plt.imshow(img)
    
    # if flows.any():
    if flows is not None:
        ax = fig.add_subplot(nb_rows,10,im+11)
        ax.set_xticks([])
        ax.set_yticks([])
        imgplot = plt.imshow(flow)
    #ax.set_title('Input Image')
    #plt.show()
    

    from PIL import Image
    # if flows is not None:
    # # if flows.any():
    #     ax = fig.add_subplot(nb_rows,10,im+21)
    # else:
    #     ax = fig.add_subplot(nb_rows,10,im+11)
        
    # ax.set_xticks([])
    # ax.set_yticks([])
    bg = Image.fromarray((255*img).astype('uint8'))
    overlay = Image.fromarray(cam_heatmap.astype('uint8'))
    blend = Image.blend(bg, overlay, 0.2)
    imgplot = plt.imshow(blend)
    #ax.set_title('Input Image with GradCAM Overlay')
    plt.tick_params(axis='both', which='both', bottom='off', left='off')
    # fig.subplots_adjust(wspace=0, hspace=0)
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def visualize_overlays(images, conv_outputs, conv_grads, flows=None):

    from skimage.transform import resize
    from matplotlib import pyplot as plt

    if flows is not None:
        nb_rows = 3
        fig_width = 23
        fig_height = 7
    else:
        nb_rows = 2
        fig_width = 22
        fig_height = 4.5

    for im in range(images.shape[1]):
        # print(im)
        image = images[0,im,:,:]
        if flows is not None:
            flow = flows[0,im,:,:]
            flow = flow.astype(float)
            flow -= np.min(flow)
            flow /= flow.max()

        output = conv_outputs[0,im,:,:]           # [7,7,512]
        grads_val = conv_grads[0,im,:,:]          # [7,7,512]
        # print("grads_val shape:", grads_val.shape)
        weights = np.mean(grads_val, axis = (0, 1)) # alpha_k, [512]
        cam = np.zeros(output.shape[0 : 2], dtype = np.float32) # [7,7]

        # Taking a weighted average
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        # Passing through ReLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam) # scale 0 to 1.0
        cam = resize(cam, (128,128), preserve_range=True)

        img = image.astype(float)
        img -= np.min(img)
        img /= img.max()

        cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
        if im == 0:
            fig = plt.figure(figsize=(fig_width, fig_height))    
        ax = fig.add_subplot(nb_rows,10,im+1)
        ax.set_xticks([])
        ax.set_yticks([])
        imgplot = plt.imshow(img)
        
        # if flows.any():
        if flows is not None:
            ax = fig.add_subplot(nb_rows,10,im+11)
            ax.set_xticks([])
            ax.set_yticks([])
            imgplot = plt.imshow(flow)
        #ax.set_title('Input Image')
        #plt.show()
        

        from PIL import Image
        if flows is not None:
        # if flows.any():
            ax = fig.add_subplot(nb_rows,10,im+21)
        else:
            ax = fig.add_subplot(nb_rows,10,im+11)
            
        ax.set_xticks([])
        ax.set_yticks([])
        bg = Image.fromarray((255*img).astype('uint8'))
        overlay = Image.fromarray(cam_heatmap.astype('uint8'))
        blend = Image.blend(bg, overlay, 0.2)
        imgplot = plt.imshow(blend)
        #ax.set_title('Input Image with GradCAM Overlay')
    plt.tick_params(axis='both', which='both', bottom='off', left='off')
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def run_sess_for_clstm_networks():
    with sess.as_default():
    
        # 2-stream
        if two_stream:
            prob = sess.run(clstm_model.preds,
                            feed_dict={images: batch_img,
                                       flows: batch_flow,
                                       K.learning_phase(): 0})
    
            print(prob)
    
            gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value =             sess.run([gb_grad, target_conv_layer, target_conv_layer_grad],
                          feed_dict={images: batch_img,
                                     flows: batch_flow,
                                     labels: batch_label,
                                     K.learning_phase(): 0})
    
            target_conv_layer_value = np.reshape(target_conv_layer_value,
                                             (1, 10, 8, 8, 32))
            target_conv_layer_grad_value = np.reshape(target_conv_layer_grad_value,
                                             (1, 10, 8, 8, 32))
        else:
            prob = sess.run(clstm_model.preds,
                            feed_dict={images: batch_img,
                                       K.learning_phase(): 0})
    
            print(prob)
            gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value =             sess.run([gb_grad, target_conv_layer, target_conv_layer_grad],
                          feed_dict={images: batch_img,
                          labels: batch_label,
                          K.learning_phase(): 0})
            
            target_conv_layer_value = np.reshape(target_conv_layer_value,
                                             (1, 10, 16, 16, 32))
            target_conv_layer_grad_value = np.reshape(target_conv_layer_grad_value,
                                             (1, 10, 16, 16, 32))
    
    
        visualize_overlays(batch_img, target_conv_layer_value,
                           target_conv_layer_grad_value)
    
        for i in range(seq_length):
            print(prob[0,i,:])
        print(batch_label)


def run_on_one_sequence(sess, clstm_model, batch_img, batch_flow, two_stream=True):
    from keras import backend as K
    with sess.as_default():
        g = sess.graph
        images = g.get_tensor_by_name("images")
        flows = g.get_tensor_by_name("flows")
        labels = g.get_tensor_by_name("labels")

        # 2-stream
        if two_stream:
            prob = sess.run(clstm_model.preds,
                            feed_dict={images: batch_img,
                                       flows: batch_flow,
                                       K.learning_phase(): 0})

            print(prob)

            gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run([gb_grad, target_conv_layer, target_conv_layer_grad],
                          feed_dict={images: batch_img,
                                     flows: batch_flow,
                                     labels: batch_label,
                                     K.learning_phase(): 0})

            target_conv_layer_value = np.reshape(target_conv_layer_value,
                                             (1, 10, 8, 8, 32))
            target_conv_layer_grad_value = np.reshape(target_conv_layer_grad_value,
                                             (1, 10, 8, 8, 32))
        else:
            prob = sess.run(clstm_model.preds,
                            feed_dict={images: batch_img,
                                       K.learning_phase(): 0})

            print(prob)
            gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run([gb_grad, target_conv_layer, target_conv_layer_grad],
                          feed_dict={images: batch_img,
                          labels: batch_label,
                          K.learning_phase(): 0})
            target_conv_layer_value = np.reshape(target_conv_layer_value,
                                         (1, 10, 16, 16, 32))
            target_conv_layer_grad_value = np.reshape(target_conv_layer_grad_value,
                                         (1, 10, 16, 16, 32))


        visualize_overlays(batch_img, target_conv_layer_value,
                           target_conv_layer_grad_value)

        for i in range(seq_length):
            print(prob[0,i,:])
        print(batch_label)

