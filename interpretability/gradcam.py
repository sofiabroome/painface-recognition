import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage.transform import resize


def get_gradcam_blend(img, image_height, image_width, cam, cam_max):
    """
    img: np.ndarray
    cam: np.ndarray
    cam_max: float
    return: Image object"""
    cam = cam / cam_max # scale 0 to 1.0
    cam = resize(cam, (image_height, image_width), preserve_range=True)

    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    # cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    bg = Image.fromarray((255*img).astype('uint8'))
    overlay = Image.fromarray(cam_heatmap.astype('uint8'))
    blend = Image.blend(bg, overlay, 0.4)

    return blend


def get_gradcam(config_dict, sess, prediction, last_clstm_output,
                y, original_input_var, mask_var, frame_inds,
                sequence, label, target_index,
                image_height, image_width):
    
    prob = tf.keras.layers.Activation('softmax')(prediction)

    # Elementwise multiplication between y and prediction, then reduce to scalar
    if target_index == np.argmax(label):
        y_c = tf.reduce_sum(tf.multiply(prediction, y), axis=1)
    else:
        # y_guessed = tf.one_hot(target_index, depth=1)
        y_guessed = tf.one_hot(tf.argmax(prob), depth=1)
        y_c = tf.reduce_sum(tf.multiply(prediction, y_guessed), axis=1)

    target_conv_layer = last_clstm_output

    # Compute gradients of class output wrt target conv layer
    target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]
    
    # Obtain values for conv and grad tensors
    target_conv_layer_value, \
        target_conv_layer_grad_value = sess.run([target_conv_layer,
                                             target_conv_layer_grad],
                                             feed_dict={original_input_var: sequence,
                                                        y: label,
                                                        mask_var: np.zeros((config_dict['seq_length'])),
                                                        frame_inds: range(config_dict['seq_length'])})
    assert (target_conv_layer_grad_value.shape[1] == config_dict['seq_length'])

    # Get CAMs (class activation mappings) for all clips
    # and save max for normalization across clip.
    cam_max = 0
    gradcams = []
    for i in range(config_dict['seq_length']):
        frame = sequence[0,i,:]
        grad = target_conv_layer_grad_value[0,i,:]
        conv_output = target_conv_layer_value[0,i,:]
        # Prepare frame for gradcam
        img = frame.astype(float)
        img -= np.min(img)
        img /= img.max()
        cam = get_cam_after_relu(conv_output=conv_output,
                                 conv_grad=grad)
        gradcams.append(cam)
        if np.max(cam) > cam_max:
            cam_max = np.max(cam)
    gradcam_masks = [] 
    # Loop over frames again to blend them with the gradcams and save.
    for i in range(config_dict['seq_length']):
        frame = sequence[0, i, :]
        # Prepare frame for gradcam
        img = frame.astype(float)
        img -= np.min(img)
        img /= img.max()
    
        # NORMALIZE PER FRAME
        if config_dict['normalization_mode'] == 'frame':
            gradcam_blend = get_gradcam_blend(img, image_height,
                    image_width, gradcams[i], np.max(gradcams[i]))
        # NORMALIZE PER SEQUENCE 
        elif config_dict['normalization_mode'] == 'sequence':
            gradcam_blend = get_gradcam_blend(img, image_height,
                    image_width, gradcams[i], cam_max)
        else:
            print('Error. Need to provide normalization mode.')
        gradcam_masks.append(gradcam_blend)
    return gradcam_masks


def get_cam_after_relu(conv_output, conv_grad):
    weights = np.mean(conv_grad, axis=(0, 1))  # alpha_k, [512]
    cam = np.zeros(conv_output.shape[0:2], dtype=np.float32)  # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * conv_output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    return cam

