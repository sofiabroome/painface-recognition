import os

import cv2
import glob
import numpy as np
from PIL import Image
import tensorflow as tf

from interpretability import mask


def visualize_results(config_dict, orig_seq, pert_seq, mask, root_dir=None,
                      case="0", mark_imgs=True, iter_test=False):
    if root_dir is None:
        root_dir = '/workspace/projects/spatiotemporal-interpretability/tensorflow/' + \
                   config_dict['output_folder'] + "/"
    root_dir += "/PerturbImgs/"

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for i in range(config_dict['seq_length']):

        if mark_imgs:
            orig_seq[:, i, :10, :10, 1:] = 0
            orig_seq[:, i, :10, :10, 0] = mask[i] * 255
            pert_seq[:, i, :10, :10, 1:] = 0
            pert_seq[:, i, :10, :10, 0] = mask[i] * 255
        result = Image.fromarray(pert_seq[0, i, :, :, :].astype(np.uint8))
        # result.save(root_dir + "case" + case + "pert" + str(i) + ".png")
        result.save(root_dir + "case" + case + "pert" + str(i) + ".jpg")
    f = open(root_dir + "case" + case + ".txt", "w+")
    f.write(str(mask))
    f.close()


def visualize_results_on_gradcam(config_dict, gradcam_images, mask, root_dir,
                                 image_width, image_height,
                                 case="0", round_up_mask=True, flow=True):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    dots = find_temp_mask_red_dots(image_width, image_height, mask, round_up_mask)

    if flow:
        dot_offset = config_dict['input_width'] * 3
        dot_offset_flow = config_dict['input_width'] * 4
    else:
        dot_offset = config_dict['input_width'] * 2

    for i in range(len(mask)):
        for j, dot in enumerate(dots):

            if i == j:
                intensity = 255
            else:
                intensity = 150

            gradcam_images[i][dot["y_start"]:, dot_offset + dot["x_start"]:dot_offset + dot["x_end"], :] = 0
            gradcam_images[i][
                dot["y_start"]:,
                dot_offset + dot["x_start"]:dot_offset + dot["x_end"],
                dot["channel"]] = intensity

            if flow:
                gradcam_images[i][dot["y_start"]:, dot_offset_flow + dot["x_start"]:dot_offset_flow + dot["x_end"], :] = 0
                gradcam_images[i][
                    dot["y_start"]:,
                    dot_offset_flow + dot["x_start"]:dot_offset_flow + dot["x_end"],
                    dot["channel"]] = intensity

            # result = Image.fromarray(gradcam_images[i].astype(np.uint8), mode="RGB")
            tmp = cv2.cvtColor(gradcam_images[i], cv2.COLOR_BGR2RGB)
            result = Image.fromarray(tmp.astype(np.uint8))
            # result.save(root_dir + "/case" + case + "_" + str(i) + ".png")
            result.save(root_dir + "/case" + case + "_" + str(i) + ".jpg")

    f = open(root_dir + "/MASKVALScase" + case + ".txt", "w+")
    f.write(str(mask))
    f.close()


def find_temp_mask_red_dots(image_width, image_height, mask, round_up_mask):
    mask_len = len(mask)
    dot_width = int(image_width // (mask_len + 4))
    dot_padding = int((image_width - (dot_width * mask_len)) // mask_len)
    dot_height = int(image_height // 20)
    dots = []

    for i, m in enumerate(mask):

        if round_up_mask:
            if mask[i] > 0.5:
                mask[i] = 1
            else:
                mask[i] = 0

        dot = {'y_start': -dot_height,
               'y_end': image_height,
               'x_start': i * (dot_width + dot_padding),
               'x_end': i * (dot_width + dot_padding) + dot_width}

        if mask[i] == 0:
            dot['channel'] = 1  # Green
        else:
            dot['channel'] = 2  # in BGR.

        dots.append(dot)

    return dots


def prepare_for_image_write(array):
    array -= np.min(array)
    array /= np.max(array)
    array = tf.cast(255*array, tf.uint8).numpy()
    return array


def create_image_arrays(config_dict, input_sequence, gradcams, time_mask,
                        output_folder, video_id, mask_type,
                        image_width, image_height):
    combined_images = []
    sequence = input_sequence[:, 0, :, :, :, :]
    perturbed_sequence = mask.perturb_sequence(
        sequence,
        time_mask,
        perturbation_type=mask_type,
        snap_values=True)
    sequence = prepare_for_image_write(sequence)

    flow_sequence = input_sequence[:, 1, :, :, :, :]
    perturbed_flow = mask.perturb_sequence(
        flow_sequence,
        time_mask,
        perturbation_type=mask_type,
        snap_values=True)
    flow_sequence = prepare_for_image_write(flow_sequence)

    perturbed_sequence = prepare_for_image_write(perturbed_sequence)
    perturbed_flow = prepare_for_image_write(perturbed_flow)
    # sequence -= np.min(sequence)
    # sequence /= np.max(sequence)
    # sequence = tf.cast(255*sequence, tf.uint8).numpy()

    for i in range(config_dict['seq_length']):
        # frame = input_sequence[0, 0, i, :, :, :]
        # frame -= np.min(frame)
        # frame /= np.max(frame)
        # frame = tf.cast(255*frame, tf.uint8).numpy()
        # frame = cv2.applyColorMap(sequence[0, i, :], cv2.COLORMAP_JET)
        frame = cv2.cvtColor(sequence[0, i, :], cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        flow = cv2.cvtColor(flow_sequence[0, i, :], cv2.COLOR_BGR2RGB)
        flow = Image.fromarray(flow)

        # combined_img = np.concatenate((np.uint8(frame),
        perturbed_frame = cv2.cvtColor(perturbed_sequence[0, i, :], cv2.COLOR_BGR2RGB)
        perturbed_frame = Image.fromarray(perturbed_frame)

        perturbed_flow_frame = cv2.cvtColor(perturbed_flow[0, i, :], cv2.COLOR_BGR2RGB)
        perturbed_flow_frame = Image.fromarray(perturbed_flow_frame)

        combined_img = np.concatenate((frame,
                                       flow,
                                       np.uint8(gradcams[i]),
                                       perturbed_frame,
                                       perturbed_flow_frame),
                                      axis=1)

        combined_images.append(combined_img)
        # cv2.imwrite(os.path.join(
        #     output_folder,
        #     "img%02d.jpg" % (i + 1)),
        #     combined_img)

    visualize_results_on_gradcam(config_dict,
                                 combined_images,
                                 time_mask,
                                 output_folder,
                                 image_width,
                                 image_height,
                                 case=mask_type + video_id)

    path_to_combined_gif = os.path.join(output_folder, "mygif.gif")
    fp_in = os.path.join(output_folder, '*.jpg')
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=path_to_combined_gif, format='GIF', append_images=imgs,
             save_all=True, duration=1000, loop=0)
    # os.system("convert -delay 10 -loop 0 {}.jpg {}".format(
    #     os.path.join(output_folder, "*"),
    #     path_to_combined_gif))

    return combined_images
