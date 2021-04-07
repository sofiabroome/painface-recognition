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
    img_255 = tf.cast(255*img, tf.uint8).numpy()
    bg = Image.fromarray(img_255)
    overlay = Image.fromarray(cam_heatmap.astype('uint8'))
    blend = Image.blend(bg, overlay, 0.4)

    return blend


def get_gradcam(config_dict, model, input_var, target_index,
                image_height, image_width):
    with tf.GradientTape() as tape:
        after_softmax, conv_output = model(input_var)
        y_c = after_softmax[:, target_index]

    gradients = tape.gradient(y_c, conv_output)

    if config_dict['model'] == 'i3d_stream':
        config_dict['seq_length'] = gradients.shape[1]
    # else:
    #     assert (gradients.shape[1] == config_dict['seq_length'])

    # Get CAMs (class activation mappings) for all clips
    # and save max for normalization across clip.
    cam_max = 0
    gradcams = []
    for i in range(config_dict['seq_length']):
        frame = input_var[0, 0, i, :]
        grad = gradients[0, i, :]
        conv_output_for_frame = conv_output[0, i, :]
        # Prepare frame for gradcam
        frame -= np.min(frame)
        frame /= np.max(frame)
        cam = get_cam_after_relu(conv_output=conv_output_for_frame,
                                 conv_grad=grad)
        gradcams.append(cam)
        if np.max(cam) > cam_max:
            cam_max = np.max(cam)
    gradcam_masks = []

    # Loop over frames again to blend them with the gradcams and save.
    for i in range(config_dict['seq_length']):
        frame = input_var[0, 0, i, :]
        # Prepare frame for gradcam
        frame -= np.min(frame)
        frame /= np.max(frame)
    
        # NORMALIZE PER FRAME
        if config_dict['normalization_mode'] == 'frame':
            gradcam_blend = get_gradcam_blend(frame, image_height,
                    image_width, gradcams[i], np.max(gradcams[i]))
        # NORMALIZE PER SEQUENCE 
        elif config_dict['normalization_mode'] == 'sequence':
            gradcam_blend = get_gradcam_blend(frame, image_height,
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

